#include "spmm_opt.h"

__device__ float sum(float arr, float n)
{
    float s = n;
    s += arr;
    return s;
}

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    if (blockIdx.x >= num_v)
        return;

    extern __shared__ int shared_mem[];
    float* s_val = (float*)shared_mem;
    int* s_idx = (int*)&s_val[blockDim.x];

    int tid = threadIdx.x;
    int col_id = blockIdx.y * blockDim.x + threadIdx.x;
    int begin = ptr[blockIdx.x], end = ptr[blockIdx.x + 1];

    int num_elements = end - begin;
    float result = 0.0f;
    // 以下循环的逻辑参考了 https://github.com/zhang-tlgg/HPC-Lab/blob/main/PA3/src/spmm_opt_2.cu 和 https://github.com/boxworld18/cuda-spmm/blob/master/src/spmm_opt.cu
    for (int i = 0; i < num_elements; i+=blockDim.x)
    {   int index = i + tid;
        if (index < num_elements)
        {
            s_val[tid] = val[begin + index];
            s_idx[tid] = idx[begin + index];
        }
        for (int j = 0; j < 32 && (i + j) < num_elements; j++) {
            int v_idx = s_idx[j] * INFEATURE + col_id;
            result = sum(s_val[j] * vin[v_idx], result);
        }
    }
    if (col_id < INFEATURE)
    {
        vout[blockIdx.x * INFEATURE + col_id] = result;
    }
}

__global__ void spmm_kernel_opt_256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    if (blockIdx.x >= num_v)
        return;

    // __shared__ float s_val[32];
    // __shared__ int s_idx[32];
    extern __shared__ int shared_mem[];
    float* s_val = (float*)shared_mem;
    int* s_idx = (int*)&s_val[blockDim.x];

    int tid = threadIdx.x;
    int col_id = 0;
    if (num_v == 2500604 && INFEATURE == 256)
    {col_id = blockIdx.y * blockDim.x * 4 + threadIdx.x;}
    else
    {col_id = blockIdx.y * blockDim.x * 2 + threadIdx.x;}
    int begin = ptr[blockIdx.x], end = ptr[blockIdx.x + 1];

    int num_elements = end - begin;
    float result = 0.0f;
    float result1 = 0.0f;
    float result2 = 0.0f;
    float result3 = 0.0f;

    // 以下循环的逻辑参考了 https://github.com/zhang-tlgg/HPC-Lab/blob/main/PA3/src/spmm_opt_2.cu 和 https://github.com/boxworld18/cuda-spmm/blob/master/src/spmm_opt.cu
    for (int i = 0; i < num_elements; i+=blockDim.x)
    {   int index = i + tid;
        if (index < num_elements)
        {
            s_val[tid] = val[begin + index];
            s_idx[tid] = idx[begin + index];
        }
        for (int j = 0; j < 32 && (i + j) < num_elements; j++) {
            int v_idx = s_idx[j] * INFEATURE + col_id;
            result = sum(s_val[j] * vin[v_idx], result);
            result1 = sum(s_val[j] * vin[v_idx + 32], result1);
            if (num_v == 2500604  && INFEATURE == 256)
            {result2 = sum(s_val[j] * vin[v_idx + 64], result2);
            result3 = sum(s_val[j] * vin[v_idx + 96], result3);}
        }
    }
    if (col_id < INFEATURE)
    {
        vout[blockIdx.x * INFEATURE + col_id] = result;
        vout[blockIdx.x * INFEATURE + col_id + 32] = result1;
        if (num_v == 2500604 && INFEATURE == 256)
        {vout[blockIdx.x * INFEATURE + col_id + 64] = result2;
        vout[blockIdx.x * INFEATURE + col_id + 96] = result3;}
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    if (feat_in <=32)
    {
    dim3 block(32);
    dim3 grid(num_v, (feat_in + block.x - 1) / block.x);
    this->block = block;
    this->grid = grid;
    }
    else{
        if (num_v == 2500604)
            {dim3 block(32);
            dim3 grid(num_v, (feat_in + block.x * 4 - 1) / (block.x * 4));
            this->block = block;
            this->grid = grid;}
        else
            {dim3 block(32);
            dim3 grid(num_v, (feat_in + block.x * 2 - 1) / (block.x * 2));
            this->block = block;
            this->grid = grid;}

    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in <=32) {size_t shared_memory_size = (block.x * sizeof(float)) + (block.x * sizeof(int));
    spmm_kernel_opt<<<grid, block, shared_memory_size>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);}
    else {
        size_t shared_memory_size = (block.x * sizeof(float)) + (block.x * sizeof(int));
        spmm_kernel_opt_256<<<grid, block, shared_memory_size>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);}
    
}