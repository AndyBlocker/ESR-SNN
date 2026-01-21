import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils import grad_scale, floor_pass, clip
from snn.nvtx import nvtx_range


class MyQuan(nn.Module):
    def __init__(self,
                 level,
                 sym=False,
                 cal_loss=False,
                 channel_num=768,
                 use_checkpoint=False,   # 新增：是否对这个量化层用 checkpoint
                 **kwargs):
        super(MyQuan, self).__init__()

        self.s_init = 0.0
        self.level = level
        self.sym = sym
        self.channel_num = channel_num
        self.use_checkpoint = use_checkpoint

        # ---------- 量化范围：用 buffer 存，随着 module.cuda() 自动迁移 ----------
        if level >= 512:
            print("level", level)
            self.pos_max = 'full'
            self.neg_min = None
            # 占位 buffer，实际不会用到
            self.register_buffer("pos_max_buf", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer("neg_min_buf", torch.tensor(0.0, dtype=torch.float32))
        else:
            print("level", level)
            if sym:
                pos = float(level // 2 - 1)
                neg = float(-level // 2 + 1)
            else:
                pos = float(level // 2 - 1)
                neg = float(0.0)

            # 这两个用于 __repr__ / debug
            self.pos_max = pos
            self.neg_min = neg

            # 真正计算用的是 buffer，会自动随 .to(device) 迁移
            self.register_buffer("pos_max_buf", torch.tensor(pos, dtype=torch.float32))
            self.register_buffer("neg_min_buf", torch.tensor(neg, dtype=torch.float32))

        # ---------- 参数：FP32 存储，方便优化器更新 ----------
        self.s = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.batch_init = 20
        self.s_max = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32),
                                  requires_grad=False)
        self.bias_channel = nn.Parameter(
            torch.zeros(self.channel_num, dtype=torch.float32),
            requires_grad=True
        )

        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"
        self.record = True
        self.cal_loss = cal_loss
        self.print_dist = False

    def __repr__(self):
        return (f"MyQuan(level={self.level}, sym={self.sym}, "
                f"pos_max={self.pos_max_buf.data.item()}, neg_min={self.neg_min_buf.data.item()}, "
                f"s={self.s.data}, s_max={self.s_max.data})")

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self, name, tfwriter, global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    # ---------------------- 内部：初始化 s（不建图） ---------------------- #
    def _init_scale(self, x_for_init):
        if (not self.training) or self.init_state != 0 or self.pos_max == 'full':
            return False

        with torch.no_grad():
            mean_abs = x_for_init.detach().abs().mean().to(torch.float32)
            Q = float(abs(self.pos_max))
            if self.sym:
                s_val = mean_abs * 2.0 / math.sqrt(Q)
            else:
                s_val = mean_abs * 4.0 / math.sqrt(Q)

            s_val = torch.clamp(s_val, min=1e-3)
            self.s.data.copy_(s_val)
            self.init_state = 1

        return True

    # ---------------------- 内部：量化核心逻辑 ---------------------- #
    def _quant_core(self, x):
        """
        x: 量化计算用的张量（可能是 BF16 / FP16 / FP32），dtype 由外面控制
        返回:
          output: 量化后的结果（和 x 同 dtype）
          x_before: 加完 bias、reshape 之后、量化之前的激活（debug 用）
          q_int: clamp 后的“整数域”值（debug 用）
        """
        with nvtx_range("snn.layer.quant.MyQuan._quant_core"):
            dtype = x.dtype
            device = x.device

            # 范围 cast 到 x 的 dtype，避免 dtype 提升到 FP32
            min_val = self.neg_min_buf.to(device).to(dtype)
            max_val = self.pos_max_buf.to(device).to(dtype)

            # LSQ 的 grad scale：用 python float，完全不进计算图
            Q = float(abs(self.pos_max))
            s_grad_scale = 1.0 / math.sqrt(Q * x.numel())

            # grad_scale 返回的本质还是 self.s / self.bias_channel，只是带了自定义梯度
            # 这里显式 cast 到 x 的 dtype，避免激活被抬到 FP32
            s_scale = grad_scale(self.s, s_grad_scale).to(dtype)
            bias_scale = grad_scale(self.bias_channel, s_grad_scale).to(dtype)

            # 加通道 bias
            if x.dim() == 3:
                x_q = x + bias_scale.view(1, 1, -1)
            elif x.dim() == 4 and x.shape[-1] != x.shape[-2]:
                B, Head, N, C = x.shape
                tmp = x.transpose(1, 2).reshape(B, N, Head * C) \
                      + bias_scale.view(1, 1, -1)
                x_q = tmp.reshape(B, N, Head, C).transpose(1, 2)
            else:
                x_q = x

            # 用乘以 1/s_scale 替代除法，少一个大张量中间结果
            inv_s = 1.0 / s_scale
            q = floor_pass(x_q * inv_s + 0.5)

            q_clamped = torch.clamp(q, min=min_val, max=max_val)
            # print("clamp min_val:",min_val, "max_val", max_val)
            output = q_clamped * s_scale

            return output, x_q, q_clamped

    # ---------------------- 真正的 forward 实现 ---------------------- #
    def _forward_impl(self, x):
        with nvtx_range("snn.layer.quant.MyQuan._forward_impl"):
            input_dtype = x.dtype

            # level >= 512：保持原行为，不量化
            if self.pos_max == 'full':
                return x

            # 1. 初始化 s（只在第一次训练时做一次）
            if self.training and self.init_state == 0:
                if input_dtype == torch.float16:
                    x_init = x.to(torch.bfloat16)
                else:
                    x_init = x
                did_init = self._init_scale(x_init)
                if did_init:
                    return x

            # 2. 正式量化
            if self.training and input_dtype == torch.float16:
                # 和你原来一样：FP16 训练时内部用 BF16 计算，避免溢出
                with torch.amp.autocast(device_type='cuda',
                                        dtype=torch.bfloat16,
                                        enabled=True):
                    x_bf16 = x.to(torch.bfloat16)
                    out_bf16, x_before, q_int = self._quant_core(x_bf16)
                output = out_bf16.to(input_dtype)
            else:
                # 推理 or 非 FP16 的情况，用当前 dtype 计算
                output_raw, x_before, q_int = self._quant_core(x)
                output = output_raw.to(input_dtype)

            # 3. TensorBoard 统计（不进计算图）
            if self.debug and self.tfwriter is not None:
                with torch.no_grad():
                    self.tfwriter.add_histogram(
                        tag="before_quan/" + self.name + "_data",
                        values=x_before.detach().cpu(),
                        global_step=self.global_step
                    )
                    self.tfwriter.add_histogram(
                        tag="after_quan/" + self.name + "_data",
                        values=q_int.detach().cpu(),
                        global_step=self.global_step
                    )

                self.debug = False
                self.tfwriter = None
                self.name = ""
                self.global_step = 0.0

            # 4. 额外 loss（和原逻辑等价）
            if self.cal_loss:
                s_val = self.s.detach().to(output.dtype)
                self.l1_loss = (output.abs() / s_val).sum() * 1e-8

            return output

    # ---------------------- 对外 forward：加可选 checkpoint ---------------------- #
    def forward(self, x):
        with nvtx_range("snn.layer.quant.MyQuan.forward"):
            # checkpoint 只在训练时有意义
            if self.use_checkpoint and self.training:
                return checkpoint(self._forward_impl, x)
            else:
                return self._forward_impl(x)

class MyQuanRound(nn.Module):
    def __init__(self,level,sym = False, **kwargs):
        super(MyQuanRound,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 512:
            print("level",level)
            self.pos_max = 'full'
        else:
            print("level",level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2 + 1))
            else:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0)).to(torch.float32)
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"
        self.record = True

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self,name,tfwriter,global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        with nvtx_range("snn.layer.quant.MyQuanRound.forward"):
            input_detype = x.dtype
            if self.pos_max == 'full':
                return x
            if str(self.neg_min.device) == 'cpu':
                self.neg_min = self.neg_min.to(x.device)
            if str(self.pos_max.device) == 'cpu':
                self.pos_max = self.pos_max.to(x.device)
            min_val = self.neg_min
            max_val = self.pos_max

            # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
            s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)

            if self.init_state == 0 and self.training:
                self.s.data = (torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max ** 0.5)).cuda() if self.sym \
                                else torch.tensor(x.detach().abs().mean() * 4 / (self.pos_max ** 0.5)).cuda())
                self.init_state += 1
                print("myquanRound init")
                return x

            self.s.data = min(torch.tensor(1/self.pos_max,device=x.device), self.s.data)
            s_scale = grad_scale(self.s, s_grad_scale)
            output = torch.clamp(torch.round(x/s_scale), min=min_val, max=max_val)*s_scale

            if self.debug and self.tfwriter is not None:
                self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
                self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
                self.debug = False
                self.tfwriter = None
                self.name = ""
                self.global_step = 0.0

            output = output.to(input_detype)
            # print("MyQuan output.abs().mean()",output.abs().mean(),output.dtype)

            # x_abs = torch.abs(output)/self.s
            # self.l2_loss = l2_loss1 + (x_abs - (1/147)*x_abs*x_abs*x_abs).sum()
            # self.absvalue = (torch.abs(output)/self.s).sum()
            # output = floor_pass(x/s_scale)*s_scale
            # print(output.abs().mean(), self.s.data.item())
            return output

class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        with nvtx_range("snn.layer.quant.QuanConv2d.forward"):
            quantized_weight = self.quan_w_fn(self.weight)
            return self._conv_forward(x, quantized_weight, self.bias)

class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        with nvtx_range("snn.layer.quant.QuanLinear.forward"):
            quantized_weight = self.quan_w_fn(self.weight)
            return torch.nn.functional.linear(x, quantized_weight, self.bias)
