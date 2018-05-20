//기본 코드 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


//host에서 호출가능하며 Device에서 실행되는 함수 커널함수 정의
__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x; // kernel을 실행할 각 thread에게는 thread ID가 주어지는데, kernel 함수 내에서 built-in variable인 ‘threadIdx’로 액세스
	c[i] = a[i] + b[i];
	printf("%d\n", i);
}

//host에서만 호출가능하며 host에서 실행되는 호스트 함수 정의
__host__ float hostFuncion()
{
	printf("hostFuncion called\n");
	return 0;
}


int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	hostFuncion();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) //결과를 넣을 벡터, 벡터1, 벡터2, 벡터 사이즈
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	//멀티 GPU 시스템 환경에서 실행할 GPU를 선택하는 코드. 
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// Allocate GPU buffers for three vectors (two input, one output).
	// Device(GPU)의 grid에 있는 Global Memory에 3개의 벡터를 위한 GPU버퍼를 할당한다.
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	// 호스트에 존재하는 버퍼를 Device(GPU)에 존재하는 GPU버퍼들로 복사한다.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); //cudaMemcpy는 비동기전송으로 작동하며 HostToHost, HostToDevice, DeviceToHost, DeviceToDevice  4가지 타입이 가능함.
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	// 각 요소에 대해 하나의 스레드로 GPU에서 커널 시작
	// N개의 thread들이 각 data pair에 대하여 한 번씩의 addKernel( )를 수행
	addKernel << <1, size >> >(dev_c, dev_a, dev_b); // 1은 생성할 쓰레드 블록 의 수, size는 블록당 쓰레드 수


													 // Check for any errors launching the kernel
													 // 커널을 시작하는 동안 에러가 있었는지 확인한다.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	// cudaDeviceSynchronize는 커널이 끝마칠 때 까지 기다린다. 그리고 그 실행동안에 발생했던 모든 오류를 반환한다.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	// GPU buffer에서 호스트 메모리로 결과벡터를 복사한다.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
