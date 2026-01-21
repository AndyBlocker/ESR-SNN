import torch


def theta_backward(x):
    sigmoid = torch.sigmoid(4*x)
    return 4*sigmoid*(1-sigmoid)

def theta(x):
    # return (x > 0).int()
    return 1.0*(torch.gt(x,0))

def theta_eq(x):
    # return (x >= 0).int()
    return 1.0*(torch.ge(x,0))

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def set_init_false(model):
    from .attention import QAttention_without_softmax
    from .quant import MyQuan, MyQuanRound
    from .window_attention import QWindowAttention

    def set_init_false_inner(model):
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, QAttention_without_softmax) or isinstance(child, QWindowAttention):
                child.init = True
            if isinstance(child, MyQuan) or isinstance(child, MyQuanRound):
                child.init_state = child.batch_init
                child.s.data = child.s.data * child.pos_max_buf.data / child.pos_max
                child.s_max.data = child.s_max.data * child.pos_max_buf.data / child.pos_max
                device = child.pos_max_buf.device
                child.pos_max_buf.data = torch.tensor(child.pos_max).to(device)
                child.neg_min_buf.data = torch.tensor(child.neg_min).to(device)
            else:
                set_init_false_inner(child)

    set_init_false_inner(model)

def cal_overfire_loss(model):
    from .st_bifneuron_ms import ST_BIFNeuron_MS

    l2_loss = 0.0

    def l2_regularization_inner(model):
        nonlocal l2_loss
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, ST_BIFNeuron_MS):
                l2_loss = l2_loss + child.overfireLoss
            else:
                l2_regularization_inner(child)

    l2_regularization_inner(model)
    return l2_loss

def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()
