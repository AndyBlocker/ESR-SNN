import torch

from snn.layer import (
    IFNeuron,
    LLConv2d,
    LLLinear,
    MyBatchNorm1d_SS,
    SDyHT_SS,
    SpikeMaxPooling,
    SpikeMaxPooling_SS,
    Spiking_LayerNorm,
    Spiking_LayerNorm_SS,
    ST_BIFNeuron_MS,
    ST_BIFNeuron_SS,
    SWindowAttention,
    SWindowAttention_SS,
    spiking_softmax_ss,
)


_RESET_TYPES = (
    IFNeuron,
    SpikeMaxPooling_SS,
    MyBatchNorm1d_SS,
    LLConv2d,
    LLLinear,
    SWindowAttention_SS,
    SWindowAttention,
    Spiking_LayerNorm,
    Spiking_LayerNorm_SS,
    SpikeMaxPooling,
    SDyHT_SS,
    ST_BIFNeuron_MS,
    ST_BIFNeuron_SS,
    spiking_softmax_ss,
)

_WORK_TYPES = (IFNeuron, LLLinear, LLConv2d)


def get_subtensors(tensor, mean, std, sample_grain=255, time_step=4):
    time_step = int(time_step)
    if time_step <= 0:
        return tensor.new_zeros((0,) + tensor.shape)
    scaled = tensor / sample_grain
    accu = tensor.new_zeros((time_step,) + tensor.shape)
    valid = min(time_step, int(sample_grain))
    if valid > 0:
        accu[:valid] = scaled
    return accu


def _collect_reset_modules(model):
    modules = []

    def _walk(module):
        for child in module.children():
            if isinstance(child, _RESET_TYPES):
                modules.append(child)
            else:
                _walk(child)

    _walk(model)
    return modules

def reset_model(model):
    reset_modules = getattr(model, "_snn_reset_modules", None)
    if reset_modules is None:
        reset_modules = _collect_reset_modules(model)
        setattr(model, "_snn_reset_modules", reset_modules)
    for module in reset_modules:
        module.reset()


class Judger:
    def __init__(self):
        self.network_finish = True
        self._work_modules = None
        self._work_modules_owner = None

    def _collect_work_modules(self, model):
        modules = []

        def _walk(module):
            for child in module.children():
                if isinstance(child, _WORK_TYPES):
                    modules.append(child)
                else:
                    _walk(child)

        _walk(model)
        self._work_modules = modules
        self._work_modules_owner = id(model)

    def judge_finish(self, model):
        if self._work_modules is None or self._work_modules_owner != id(model):
            self._collect_work_modules(model)
        self.network_finish = all(not module.is_work for module in self._work_modules)

    def reset_network_finish_flag(self):
        self.network_finish = True
