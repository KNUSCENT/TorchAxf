import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_2_t
from typing import Union
from . import custom_functional

from torch.utils.cpp_extension import load

class fn_A_Conv2d(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, stride, cuda_stride, groups, cuda_groups , padding, f, s, cuda_output, cuda_output_size, cuda_input_size, cuda_weight_size, output_size, bias=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.stride = stride
        ctx.groups = groups
        return custom_functional.conv2d(input, weight, cuda_stride, cuda_groups, padding, f, s, cuda_output, cuda_output_size, cuda_input_size, cuda_weight_size, output_size)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables   
        grad_input = grad_weight= grad_bias = None
        padding = ctx.padding
        stride = ctx.stride
        groups = ctx.groups
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride=stride, padding=padding, groups=groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
    
        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None

class CPU_fn_A_Conv2d(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, kernel_size, stride, padding, groups, custom, bias=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.stride = stride
        ctx.groups = groups
        return custom_functional.cpu_conv2d(input, weight, kernel_size, stride, groups, padding, custom)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables   
        grad_input = grad_weight= grad_bias = None
        padding = ctx.padding
        stride = ctx.stride
        groups = ctx.groups
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride=stride, padding=padding, groups=groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
    
        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight, None, None, None, None

class A_Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.fn = None
        
        # make cuda function and stream
        self.f, self.s = custom_functional.make_cuda_function()
        # pre define parameter
        self.output:Tensor
        self.cuda_output_size:Tensor
        self.cuda_input_size:Tensor
        self.cuda_weight_size:Tensor
        self.cuda_stride:Tensor
        self.cuda_groups:Tensor
        self.output_size = []
        self.n_f = None
        self.out_h = None
        self.out_w = None
        self.col:Tensor
        self.mm_out:Tensor
        # check parameter
        self.batch_size = None

        # CPU conv2d
        self.custom = load(
            name='custom',
            sources=['./models/custom/custom_convolution.cpp'],
            extra_cflags=['-march=native -fopenmp -O3'],
            extra_ldflags=['-lgomp'],
            verbose=False
        )

        super(A_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def GPU_forward(self, input: Tensor) -> Tensor:
        n, c, h, w = input.size()
        if self.batch_size == None:
            self.fn = fn_A_Conv2d.apply
            self.batch_size = n
            n, c, h, w = input.size()
            input_t = F.pad(input, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), "constant", 0)

            n_f, c2, filter_h, filter_w = self.weight.size()
            out_h = (h + 2 * self.padding[0] - filter_h) // self.stride[0] + 1 
            out_w = (w + 2 * self.padding[1] - filter_w) // self.stride[1] + 1
            
            self.n_f = n_f
            self.out_h = out_h
            self.out_w = out_w

            input_size = input_t.size()
            weight_size = [n_f, c2, filter_h, filter_w]
            output_size = [n, n_f, out_h, out_w]
            self.output_size = output_size
            self.cuda_stride = torch.tensor(self.stride[0]).int().cuda()
            self.cuda_groups = torch.tensor(self.groups).int().cuda()
            self.output = torch.zeros(n, n_f, out_h, out_w).cuda()
            self.cuda_output_size = torch.tensor(output_size).int().cuda()
            self.cuda_input_size = torch.tensor(input_size).int().cuda()
            self.cuda_weight_size = torch.tensor(weight_size).int().cuda()

        if self.batch_size != n:
            self.output_size = [n, self.n_f, self.out_h, self.out_w]
            self.output.resize_(n, self.n_f, self.out_h, self.out_w)
            
        return self.fn(input, self.weight, self.stride, self.cuda_stride, self.groups, self.cuda_groups , self.padding, self.f, self.s, self.output, self.cuda_output_size, self.cuda_input_size, self.cuda_weight_size, self.output_size)

    def CPU_forward(self, input: Tensor) -> Tensor:
        if self.fn == None:
            self.fn = CPU_fn_A_Conv2d.apply
        return self.fn(input, self.weight, self.kernel_size, self.stride, self.padding, self.groups, self.custom)

    def forward(self, input: Tensor) -> Tensor:
        if input.get_device() == 0:
            return self.GPU_forward(input)
        else:
            return self.CPU_forward(input)
