import torch
import torch.nn.functional as F
from collections import namedtuple
from cupy.cuda import function
from pynvrtc.compiler import Program

def make_cuda_function():
    kernel = ""
    with open('./models/custom/custom_conv2d.cu','r') as f:
        kernel = f.read()
    program = Program(kernel, 'custom_conv2d.cu')
    ptx = program.compile()
    m = function.Module()
    m.load(bytes(ptx.encode()))
    f = m.get_function('conv2d')
    Stream = namedtuple('Stream', ['ptr'])
    s = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    return f, s

def conv2d(input, weight, stride, groups, padding, f, s, cuda_output, cuda_output_size, cuda_input_size, cuda_weight_size, output_size):
    input = F.pad(input, (padding[1], padding[1], padding[0], padding[0]), "constant", 0)
    f(grid=(output_size[0], output_size[1], 1), block=(output_size[2], output_size[3], 1), args=[cuda_output.data_ptr(), input.data_ptr(), weight.data_ptr(), cuda_input_size.data_ptr(), cuda_weight_size.data_ptr(), cuda_output_size.data_ptr(), stride.data_ptr(), groups.data_ptr()], stream=s)
    return cuda_output