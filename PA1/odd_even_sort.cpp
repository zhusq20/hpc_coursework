#include "worker.h"
#include <limits.h>
#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

// 计算天花板值，即a除以b向上取整
size_t ceiling(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// 计算左邻居的进程排名
size_t dec_rank_left(size_t rank) {
    return  (rank % 2 == 0) ? rank - 1 : rank + 1;
}

// 计算右邻居的进程排名
size_t dec_rank_right(size_t rank) {
    return  (rank % 2 == 0) ? rank + 1 : rank - 1;
}

int judge(int rk, int nprocs)
{   if (rk == -1 || rk == nprocs)
        rk = MPI_PROC_NULL;
    return rk;
}

// 合并两个有序数组
#include <cstddef> // for size_t
#include <mpi.h>   // for MPI_Status

// 合并两个有序数组
void merging_function(float *data, float *localBuffer, float *partner_buffer, size_t block_len, size_t partner_len, int rank, MPI_Status status, int size) 
{
    if (status.MPI_SOURCE >= 0 && status.MPI_SOURCE < size) {
        bool is_lower_rank = rank < status.MPI_SOURCE;
        size_t i = is_lower_rank ? 0 : block_len - 1;
        size_t j = is_lower_rank ? 0 : partner_len - 1;
        size_t k = is_lower_rank ? 0 : block_len - 1;

        while ((is_lower_rank && k < block_len) || (!is_lower_rank && k != static_cast<size_t>(-1))) {
            if (is_lower_rank) {
                if (j == partner_len || (i < block_len && localBuffer[i] < partner_buffer[j])) {
                    data[k++] = localBuffer[i++];
                } else {
                    data[k++] = partner_buffer[j++];
                }
            } else {
                if (j == static_cast<size_t>(-1) || (i != static_cast<size_t>(-1) && localBuffer[i] >= partner_buffer[j])) {
                    data[k--] = localBuffer[i--];
                } else {
                    data[k--] = partner_buffer[j--];
                }
            }
        }
    }
}


// Worker类的排序函数
void Worker::sort() {
    // 如果超出范围则返回
    if (out_of_range) {
        return;
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 获取MPI进程总数
    MPI_Status status; // 定义MPI状态变量
    float * localBuffer;
    size_t block_size = ceiling(n, nprocs); // 计算块大小
    float * partner_buffer = new float[block_size]; // 分配缓冲区
    int neighborRank[2];

    std::sort(data, data + block_len); // 对数据进行排序

    neighborRank[0] = dec_rank_left(rank); // 计算左邻居排名
    neighborRank[1] = dec_rank_right(rank); // 计算右邻居排名
    MPI_Request request[2];
    for (int i = 0; i < 2; ++i) {
        // 检查邻居排名是否有效
        neighborRank[i] = judge(neighborRank[i], nprocs);
    }
    
    // 分配本地缓冲区
    localBuffer = new float[block_len];
    for (int p = 0; p < nprocs - 1; p++) {

        int partner_rank = neighborRank[1 - p % 2]; // 确定合作者排名
        int partner_len;

        // 非阻塞发送和接收消息
        MPI_Isend(data, block_len, MPI_FLOAT, partner_rank, p, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(partner_buffer, block_size, MPI_FLOAT, partner_rank, p, MPI_COMM_WORLD, &request[1]);
        MPI_Wait(&request[1], &status); // 等待接收完成
        MPI_Get_count(&status, MPI_FLOAT, &partner_len); // 获取接收数据长度

        std::copy(data, data + block_len, localBuffer); // 复制数据到本地缓冲区

        merging_function(data, localBuffer, partner_buffer, block_len, partner_len, rank, status, size);
        MPI_Wait(&request[0], nullptr); // 等待发送完成
    }
    delete[] partner_buffer;
    delete[] localBuffer;
}
