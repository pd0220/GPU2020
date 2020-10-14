// including used headers
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

// kernel
__global__ void adjacent_difference(int n, float *x)
{
	// data indices to blocks
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// run algorithm
	// first element ~ do nothing
	if (i == 0)
		x[i] = x[i];
	// rest of the elements ~ compute differences
	if (i < n)
	{
		x[i] = x[i] - x[i - 1];
	}
}

// main function
int main(int, char **)
{
	// test vectors
	std::vector<float> XVec{2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f, 16.f, 18.f, 20.f};
	std::vector<float> ResultVec(XVec.size());

	// sizes
	size_t size = XVec.size();
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
	dim3 dimGrid(1);
	dim3 dimBlock(static_cast<int>(size));

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
	for (auto r : ResultVec)
	{
		std::cout << r << std::endl;
	}

	// repeat on CPU tp validate results
	std::adjacent_difference(XVec.begin(), XVec.end(), XVec.begin());

	// check equality
	if (std::equal(ResultVec.begin(), ResultVec.end(), XVec.begin()))
	{
		std::cout << "Success" << std::endl;
	}
	else
	{
		std::cout << "Mismatch between CPU and GPU results." << std::endl;;
	}

	// this is the way
	return 0;
}