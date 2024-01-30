import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _single
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t
from typing import Union 

from . import custom_functional

class fn_A_Conv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=False):
        ctx.save_for_backward(input, weight, bias)
        stride = 1
        padding = 0
        ctx.stride = stride
        ctx.padding = padding

#        tensor_a = custom_functional.conv1d(input, weight)
#        tensor_b = F.conv1d(input, weight, bias, stride, padding)
#
#        mismatch = tensor_a != tensor_b
#        different_values1 = tensor_a[mismatch]
#        different_values2 = tensor_b[mismatch]
#
#        print("size: ", tensor_a.size())
#        print("diffenent_valuse1:", different_values1)
#        print("diffenent_valuse2:", different_values2)
#
#        print("---------------------------------------------------")

        return custom_functional.conv1d(input, weight)
        #return F.conv1d(input, weight, bias, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv1d_input(input.shape, weight, grad_output, stride=stride, padding=padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv1d_weight(input, weight.shape, grad_output, stride=stride, padding=padding)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias

        return grad_input, grad_weight, None

class A_Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        
        self.fn = fn_A_Conv1d.apply

        super().__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor):
        return self.fn(input, weight, self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight)    
