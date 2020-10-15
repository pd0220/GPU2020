// including used headers
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

// kernel
__global__ void adjacent_difference(int n, float *x)
{
	// data indices to blocks
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// run algorithm
	// first element ~ do nothing
	// rest of the elements ~ compute differences
	if (i < n && i != 0)
	{
		x[i] = x[i] - x[i - 1];
	}
}

// main function
int main(int, char **)
{
	// random number generation
	std::random_device rd{};
	std::mt19937 gen(rd());
	std::normal_distribution<float> distr(-10.f, 10.f);

	auto rand = [&distr, &gen]() {
		return (float)distr(gen);
	};

	// size
	size_t size = (int)(1000 * 1000);

	// test vectors
	std::vector<float> XVec(size);
	std::generate(XVec.begin(), XVec.end(), rand);
	std::vector<float> ResultVec(XVec.size());

	// vectors for devcie
	float *devX = nullptr;

	// memory allocation on device
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&devX, size * sizeof(float));
	if (err != cudaSuccess)
	{
		std::cout << "Error allocating CUDA memory (X): " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// copy data to onto device
	err = cudaMemcpy(devX, XVec.data(), size * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cout << "Error copying memory to device (X): " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// grid and block dimensions
	dim3 dimGrid(size / 1000, 1);
	dim3 dimBlock(1000, 1);

	// start kernel
	adjacent_difference<<<dimGrid, dimBlock>>>((int)size, devX);

	// get errors from run
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// copy data from device
	err = cudaMemcpy(ResultVec.data(), devX, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// free memory
	err = cudaFree(devX);
	if (err != cudaSuccess)
	{
		std::cout << "Error freeing allocation (X): " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// write results to screen
	/*
	for (auto r : ResultVec)
	{
		std::cout << r << std::endl;
	}
	*/

	// repeat on CPU tp validate results
	std::adjacent_difference(XVec.begin(), XVec.end(), XVec.begin());

	// check equality
	if (std::equal(ResultVec.begin(), ResultVec.end(), XVec.begin()))
	{
		std::cout << "Success" << std::endl;
	}
	else
	{
		std::cout << "Mismatch between CPU and GPU results." << std::endl;
	}

	// this is the way
	return 0;
}