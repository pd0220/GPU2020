// given code from https://github.com/Wigner-GPU-Lab/Teaching/tree/gpgpu2-2020-1/GPGPU1/CUDA/matmul

// used headers and libraries
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include "cpu_matmul.h"

// gpu matrix block size 1
static const int MBS1 = 32;
// gpu matrix block size 2
static const int MBS2 = MBS1 / 2;

// kernel
__global__ void matmul_improved(float *C, float *A, float *B, int N)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y2 = blockIdx.y * blockDim.y * 2;
    unsigned int yT = blockIdx.y * blockDim.y * 2 + threadIdx.x;

    __shared__ float Atmp[MBS1 * MBS2];
    __shared__ float Btmp[MBS1 * MBS2];

    float sum1 = 0;
    float sum2 = 0;
    for (int bk = 0; bk < N / MBS2; ++bk)
    {
        Atmp[threadIdx.x * MBS2 + threadIdx.y] = A[yT * N + (bk * MBS2 + threadIdx.y)];
        Btmp[threadIdx.y * MBS1 + threadIdx.x] = B[(bk * MBS2 + threadIdx.y) * N + x];
        __syncthreads();
        for (int k = 0; k < MBS2; ++k)
        {
            float b = Btmp[k * MBS1 + threadIdx.x];
            sum1 += Atmp[(threadIdx.y + 0) * MBS2 + k] * b;
            sum2 += Atmp[(threadIdx.y + MBS2) * MBS2 + k] * b;
        }
        __syncthreads();
    }
    C[(y2 + threadIdx.y + 0) * N + x] = sum1;
    C[(y2 + threadIdx.y + MBS2) * N + x] = sum2;
}

// main function
int main()
{
    const int N = 1024;
    const int block_sz1 = MBS1;
    const int block_sz2 = MBS2;
    const int n_blocks1 = N / MBS1;
    const int n_blocks2 = N / MBS2 / 2;

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C0(N * N);
    std::vector<float> C1(N * N);
    std::vector<float> C2(N * N);

    // generates random integers
    std::mt19937 mersenne_engine{42};
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
    generate(B.begin(), B.end(), gen);
    std::fill(C0.begin(), C0.end(), 0.0f);
    std::fill(C1.begin(), C1.end(), 0.0f);
    std::fill(C2.begin(), C2.end(), 0.0f);

    float *pA = nullptr;
    float *pB = nullptr;
    float *pC2 = nullptr;

    cudaEvent_t evt[2];
    for (auto &e : evt)
    {
        cudaEventCreate(&e);
    }

    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&pA, N * N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc((void **)&pB, N * N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMalloc((void **)&pC2, N * N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMemcpy(pA, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaMemcpy(pB, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    {
        dim3 dimGrid(n_blocks1, n_blocks2);
        dim3 dimBlock(block_sz1, block_sz2);
        cudaEventRecord(evt[0]);
        matmul_improved<<<dimGrid, dimBlock>>>(pC2, pA, pB, N);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n";
            return -1;
        }
        cudaEventRecord(evt[1]);
    }

    err = cudaMemcpy(C2.data(), pC2, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(pA);
    if (err != cudaSuccess)
    {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(pB);
    if (err != cudaSuccess)
    {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    err = cudaFree(pC2);
    if (err != cudaSuccess)
    {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    cudaEventSynchronize(evt[1]);
    // milliseconds
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    for (auto &e : evt)
    {
        cudaEventDestroy(e);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_naive(C0, A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_matmul_improved(C1, A, B, N, MBS1);
    auto t2 = std::chrono::high_resolution_clock::now();

    const float max_err = 1e-5f;
    auto comparator = [max_err](float l, float r) { return std::abs(l - r) < max_err; };

    for (int i = 0; i < N * N; ++i)
    {
        if (!comparator(C0[i], C1[i]))
        {
            std::cout << "C0 vs C1 [" << i << "] : " << C0[i] << "   " << C1[i] << " absolute error: " << std::abs(C0[i] - C1[i]) << "\n";
        }
    }

    for (int i = 0; i < N * N; ++i)
    {
        if (!comparator(C0[i], C2[i]))
        {
            std::cout << "C0 vs C2 [" << i << "] : " << C0[i] << "   " << C2[i] << " absolute error: " << std::abs(C0[i] - C2[i]) << "\n";
        }
    }

    if (std::equal(C0.begin(), C0.end(), C1.begin(), comparator))
    {
        std::cout << "CPU improved matches CPU naive.\n";
    }
    else
    {
        std::cout << "Mismatch in the two CPU results.\n";
    }

    if (std::equal(C0.begin(), C0.end(), C2.begin(), comparator))
    {
        std::cout << "GPU improved matches CPU naive.\n";
    }
    else
    {
        std::cout << "Mismatch in CPU and GPU results.\n";
    }

    std::cout << "CPU naive    Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f << " ms\n";
    std::cout << "CPU improved Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f << " ms\n";
    std::cout << "GPU improved Computation took: " << dt << " ms.\n";
    return 0;
}