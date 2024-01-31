#include "./models/custom/custom_HW.cu"

__device__ float custom_hw(float num1, float num2, float _result)
{
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
    if(Approx_MUL8u_GTR == 1) c = custom_multiplier_GTR(a, b);
    else if(Approx_MUL8u_18UH == 1) c = custom_multiplier_18UH(a, b);
    else c = custom_multiplier(a, b);
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

__device__ float custom_HW_adder(float num1, float num2)
{
	float result = 0.0;
	sfp_add A, B, C;
#if defined(ADD_FP32)
	A.f = num1;
	B.f = num2;

	if(LOA_SIZE != 0) C = loa_fpadder(A,B);
	else if(AMA5_SIZE != 0) C = ama5_fpadder(A,B);
	else if(ETA1_SIZE != 0) C = eta1_fpadder(A,B);
	else C = custom_adder(A,B);
	result = C.f;
#else
	A = add_float_to_sfp(num1);
	B = add_float_to_sfp(num2);

	if(LOA_SIZE != 0) C = loa_fpadder(A,B);
   	else if(AMA5_SIZE != 0) C = ama5_fpadder(A,B);
    	else if(ETA1_SIZE != 0) C = eta1_fpadder(A,B);
    	else C = custom_adder(A,B);
	result = add_sfp_to_float(C);
#endif
	return result;
}

extern "C" __global__ void conv2d(float *output, float *input, float *weight, const int *input_size, const int *weight_size, const int *output_size, int *stride, int *groups)
{
    int bidx = blockIdx.x;  // batch_size
    int bidy = blockIdx.y;  // out_channels
    int tidx = threadIdx.x; // out_image_h
    int tidy = threadIdx.y; // out_image_w

    int group = *groups;

    float result = 0;

    // int in_idx = bidx * input_size[1] * input_size[2] * input_size[3];
    // int w_idx = bidy * weight_size[1] * weight_size[2] * weight_size[3];

    int group_in_channel = input_size[1] / group;
    int group_out_channel = weight_size[0] / group;
    int group_id = bidy / group_out_channel;

    int in_idx = bidx * input_size[1] * input_size[2] * input_size[3] + group_id * group_in_channel * input_size[2] * input_size[3];
    int w_idx = bidy * weight_size[1] * weight_size[2] * weight_size[3];

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

extern "C" __global__ void conv1d(float *input, float *weight, float *output, const int *output_size, const int *ch) 
{
	int tidx = threadIdx.x; // data
	int bidx = blockIdx.x;  // batch_size
	int bidy = blockIdx.y;  // out_channels

	int output_size1 = output_size[0];
	int output_size2 = output_size[1];
	int output_size3 = output_size[2];
	int ch_size = *ch;

	float result = 0.0;

	for(int i=0; i<ch_size; i++)
	{
		// result = result + (input[(bidx * ch_size * output_size3) + (i*output_size3) + tidx] * weight[(bidy * ch_size) + i]);
		result = custom_hw(input[(bidx * ch_size * output_size3) + (i*output_size3) + tidx], weight[(bidy * ch_size) + i], result);
	}

	output[(bidx * output_size2 * output_size3) + (bidy * output_size3) + tidx] = result;
	
}

extern "C" __global__ void matrix_add(float *input1, float *input2, const int *size1, const int *size2, const int *size3)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    	int z = blockIdx.z * blockDim.z + threadIdx.z;

    	if (x < *size1 && y < *size2 && z < *size3) {
        	int index = z * (*size1) * (*size2) + y * (*size1) + x;
        	//input1[index] = input1[index] - input2[index];
		input1[index] = custom_HW_adder(input1[index], -input2[index]);
		if(input1[index] < 0) input1[index] = -input1[index];
    	}

}

extern "C" __global__ void matrix_sum(float *output, float *input, const int *size1, const int *size2, const int *size3)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < (*size1) * (*size3)) {
        	int row = idx / (*size3);
        	int column = idx % (*size3);
        	float sum = 0.0;

        	for (int i = 0; i < (*size2); i++) {
            		//sum += input[row * (*size2) * (*size3) + i * (*size3) + column];
			sum = custom_HW_adder(sum, input[row * (*size2) * (*size3) + i * (*size3) + column]);
        	}

        	output[row * (*size3) + column] = sum;
    	}
}