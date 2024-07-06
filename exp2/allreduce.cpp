#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#define EPS 1e-5

namespace ch = std::chrono;


int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank) {
    int block_size = n / comm_sz; // Assume n is divisible by comm_sz
    float *temp_buf = new float[block_size];

    memcpy(recvbuf, sendbuf, n * sizeof(float));

    MPI_Status status_send, status_recv;
    MPI_Request send_req, recv_req;

    // Reduce-scatter phase
    for (int step = 0; step < comm_sz - 1; ++step) {
        int send_to = (my_rank + 1) % comm_sz;
        int recv_from = mod(my_rank - 1, comm_sz);

        int send_index = mod(my_rank - step, comm_sz);
        int recv_index = mod(my_rank - step - 1, comm_sz);

        MPI_Isend(&((float*)recvbuf)[send_index * block_size], block_size, MPI_FLOAT, send_to, 0, comm, &send_req);
        MPI_Irecv(temp_buf, block_size, MPI_FLOAT, recv_from, 0, comm, &recv_req);

        MPI_Wait(&recv_req, &status_recv);
        MPI_Wait(&send_req, &status_send); // Ensure the send operation is completed

        // Accumulate received block
        for (int i = 0; i < block_size; ++i) {
            ((float*)recvbuf)[recv_index * block_size + i] += temp_buf[i];
        }
    }

    // Allgather phase
    for (int step = 0; step < comm_sz - 1; ++step) {
        int send_to = (my_rank + 1) % comm_sz;
        int recv_from = mod(my_rank - 1, comm_sz);

        int send_index = mod(my_rank - step + 1, comm_sz);
        int recv_index = mod(my_rank - step, comm_sz);

        MPI_Isend(((float*)recvbuf) + send_index * block_size, block_size, MPI_FLOAT, send_to, 0, comm, &send_req);
        MPI_Irecv(((float*)recvbuf) + recv_index * block_size, block_size, MPI_FLOAT, recv_from, 0, comm, &recv_req);

        MPI_Wait(&recv_req, &status_recv);
        MPI_Wait(&send_req, &status_send); // Ensure the send operation is completed
    }

    free(temp_buf);
}

// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            // std::cout << "mpi_recvbuf[" << i << "] = " << mpi_recvbuf[i] << ", ring_recvbuf[" << i << "] = " << ring_recvbuf[i] << std::endl;
            correct = false;
            break;
        }
        else{
            // std::cout << "mpi_recvbuf[" << i << "] = " << mpi_recvbuf[i] << ", ring_recvbuf[" << i << "] = " << ring_recvbuf[i] << std::endl;
        }



    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
