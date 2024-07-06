#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "apsp.h"

#define TILE_SIZE 32

__device__ int get_graph_element(int n, int *graph, int row, int col) {
    return (row < n && col < n) ? graph[row * n + col] : 100001;
}

__device__ void set_graph_element(int n, int *graph, int row, int col, int value) {
    if (row < n && col < n) {
        graph[row * n + col] = value;
    }
}

__global__ void phase_one_kernel(int n, int *graph, int block_id) {
    __shared__ int pivot_block[TILE_SIZE][TILE_SIZE];
    int pivot_i = block_id * TILE_SIZE;
    int pivot_j = block_id * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load the pivot block into shared memory
    pivot_block[ty][tx] = get_graph_element(n, graph, pivot_i + ty, pivot_j + tx);
    __syncthreads();

    // Perform the Floyd-Warshall update on the pivot block
    for (int k = 0; k < TILE_SIZE; k++) {
        int new_val = pivot_block[ty][k] + pivot_block[k][tx];
        pivot_block[ty][tx] = min(pivot_block[ty][tx], new_val);
        __syncthreads();
    }

    // Write the updated pivot block back to the global memory
    set_graph_element(n, graph, pivot_i + ty, pivot_j + tx, pivot_block[ty][tx]);
}

__global__ void phase_two_kernel(int n, int *graph, int block_id) {
    __shared__ int pivot_block[TILE_SIZE][TILE_SIZE];
    __shared__ int row_block[TILE_SIZE][TILE_SIZE];
    __shared__ int col_block[TILE_SIZE][TILE_SIZE];

    int pivot_i = block_id * TILE_SIZE;
    int pivot_j = block_id * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load the pivot block into shared memory
    pivot_block[ty][tx] = get_graph_element(n, graph, pivot_i + ty, pivot_j + tx);
    __syncthreads();

    // Row-wise update
    int block_row = blockIdx.x;
    if (block_row != block_id) {
        int row_i = block_row * TILE_SIZE;
        row_block[ty][tx] = get_graph_element(n, graph, row_i + ty, pivot_j + tx);
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            int new_val = row_block[ty][k] + pivot_block[k][tx];
            row_block[ty][tx] = min(row_block[ty][tx], new_val);
        }

        __syncthreads();
        set_graph_element(n, graph, row_i + ty, pivot_j + tx, row_block[ty][tx]);
    }

    // Column-wise update
    int block_col = blockIdx.y;
    if (block_col != block_id) {
        int col_j = block_col * TILE_SIZE;
        col_block[ty][tx] = get_graph_element(n, graph, pivot_i + ty, col_j + tx);
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            int new_val = pivot_block[ty][k] + col_block[k][tx];
            col_block[ty][tx] = min(col_block[ty][tx], new_val);
        }

        __syncthreads();
        set_graph_element(n, graph, pivot_i + ty, col_j + tx, col_block[ty][tx]);
    }
}

__global__ void phase_three_kernel(int n, int *graph, int block_id) {
    __shared__ int row_block[TILE_SIZE][TILE_SIZE];
    __shared__ int col_block[TILE_SIZE][TILE_SIZE];

    int pivot_i = block_id * TILE_SIZE;
    int pivot_j = block_id * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column indices for the current block
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    int row_i = block_row * TILE_SIZE;
    int col_j = block_col * TILE_SIZE;

    // Load the row block and column block into shared memory
    row_block[ty][tx] = get_graph_element(n, graph, row_i + ty, pivot_j + tx);
    col_block[ty][tx] = get_graph_element(n, graph, pivot_i + ty, col_j + tx);
    __syncthreads();

    // Perform the Floyd-Warshall update on the current block
    int new_val = get_graph_element(n, graph, row_i + ty, col_j + tx);
    for (int k = 0; k < TILE_SIZE; k++) {
        new_val = min(new_val, row_block[ty][k] + col_block[k][tx]);
    }
    set_graph_element(n, graph, row_i + ty, col_j + tx, new_val);
}

void apsp(int n, int *graph) {
    int rounds = (n + TILE_SIZE - 1) / TILE_SIZE;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks_phase_two(rounds, 2);
    dim3 blocks_phase_three(rounds, rounds);

    for (int r = 0; r < rounds; r++) {
        phase_one_kernel<<<1, threads>>>(n, graph, r);
        cudaDeviceSynchronize();
        phase_two_kernel<<<blocks_phase_two, threads>>>(n, graph, r);
        cudaDeviceSynchronize();
        phase_three_kernel<<<blocks_phase_three, threads>>>(n, graph, r);
        cudaDeviceSynchronize();
    }
}
