#include "./models/custom/custom_HW.cu"

__device__ float custom_hw(float num1, float num2, float _result)
{
#if defined(TEST)
    return _result + num1 * num2;
#endif

    float mul_result = 0;
    sfp_mul a, b, c;
#if defined(MUL_FP32)
    a.f = num1;
    b.f = num2;
    if(Approx_MUL12_2DA == 1) c = custom_multiplier_2DA(a, b);
    else if(Approx_MUL12_2JV == 1) c = custom_multiplier_2JV(a, b);
    else c = custom_multiplier(a, b);
    mul_result = c.f;
#else
    a = mul_float_to_sfp(num1);
    b = mul_float_to_sfp(num2);
    c = custom_multiplier(a, b);
    mul_result = mul_sfp_to_float(c);
#endif  

    float add_result = 0;
    sfp_add A, B, C;
#if defined(ADD_FP32)
    A.f = _result;
    B.f = mul_result;

    if(LOA_SIZE != 0) C = loa_fpadder(A,B);
    else if(AMA5_SIZE != 0) C = ama5_fpadder(A,B);
    else if(ETA1_SIZE != 0) C = eta1_fpadder(A,B);
    else C = custom_adder(A,B);

    add_result = C.f;
#else
    A = add_float_to_sfp(_result);
    B = add_float_to_sfp(mul_result);

    if(LOA_SIZE != 0) C = loa_fpadder(A,B);
    else if(AMA5_SIZE != 0) C = ama5_fpadder(A,B);
    else if(ETA1_SIZE != 0) C = eta1_fpadder(A,B);
    else C = custom_adder(A,B);
    
    add_result = add_sfp_to_float(C);
#endif
    return add_result;
}

extern "C" __global__ void conv2d(float *output, float *input, float *weight, const int *input_size, const int *weight_size, const int *output_size, int *stride)
{
    int bidx = blockIdx.x;  // batch_size
    int bidy = blockIdx.y;  // out_channels
    int tidx = threadIdx.x; // out_image_h
    int tidy = threadIdx.y; // out_image_w

    float result = 0;

    int in_idx = bidx * input_size[1] * input_size[2] * input_size[3];
    int w_idx = bidy * weight_size[1] * weight_size[2] * weight_size[3];

    sfp_mul a, b;
    sfp_add A, B;

    for (int i = 0; i < weight_size[1]; i++)
    {
        for (int row = 0; row < weight_size[2]; row++)
        {
            for (int col = 0; col < weight_size[3]; col++)
            {
                // result += input[in_idx + (i * input_size[2] * input_size[3]) + ((tidx*(*stride) + row) * input_size[3]) + (tidy*(*stride) + col)] * weight[w_idx + (i * weight_size[2] * weight_size[3]) + (row * weight_size[3]) + (col)];
                result = custom_hw(input[in_idx + (i * input_size[2] * input_size[3]) + ((tidx * (*stride) + row) * input_size[3]) + (tidy * (*stride) + col)], weight[w_idx + (i * weight_size[2] * weight_size[3]) + (row * weight_size[3]) + (col)], result);
            }
        }
    }
    output[(bidx * output_size[1] * output_size[2] * output_size[3]) + (bidy * output_size[2] * output_size[3]) + (tidx * output_size[3]) + (tidy)] = result;
}