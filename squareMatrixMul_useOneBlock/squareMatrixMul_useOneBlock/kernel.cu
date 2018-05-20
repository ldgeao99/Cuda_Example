//GPU연산으로 처리해보는 버전
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h> //clock(), time_t타입의 변수

__global__ void squareMatrixMulKernel(int *c, int *a, int *b, int arrayWidth);	//바로 위에 선언한 함수 안에서 쓰레드 생성과 함께 호출되는 함수, host에서 호출가능하며 Device에서 실행되는 커널 함수
cudaError_t squareMatrixMulWithGPU(int *c, int *a, int *b, int arrayWidth);	// 두 정방행렬의 곱셈연산을 GPU에서 수행하는 함수
void squareMatrixMulWithCPU(int *c, int *a, int *b, int arrayWidth);		// 두 정방행렬의 곱셈연산을 CPU에서 수행하는 함수 
void initArrayToRandom(int *array, int arrayWidth);				// 랜덤한 수로 행렬을 초기화 하는 함수 
void initArrayToZero(int *array, int arrayWidth);				// 0으로 행렬을 초기화 하는 함수
void printArrayAllElement(int *array, int arrayWidth);				// 행렬의 모든 원소를 출력하는 함수

int main()
{
	const int arrayWidth = 16;	//블록 한개만을 이용해 구현하였으므로 한 블록에 최대 1024개의 쓰레드만 허용가능하고 따라서 32가 최대 허용치이다. 이를 넘어가면 GPU에서 오류를 뱉을 것이다.

	//아래에 주석처리해 놓은 방법으로도 선언 가능
	int a[arrayWidth*arrayWidth] = { 0 };
	int b[arrayWidth*arrayWidth] = { 0 };
	int c[arrayWidth*arrayWidth] = { 0 };

	/*
	int *a;
	int *b;
	int *c;

	a = (int*)malloc(sizeof(int)*arrayWidth*arrayWidth);
	b = (int*)malloc(sizeof(int)*arrayWidth*arrayWidth);
	c = (int*)malloc(sizeof(int)*arrayWidth*arrayWidth);
	*/

	//이 함수가 메인함수에서 작동해야 같은 수가 안나오게됨.
	srand(time(NULL));

	//연산을 시작하기전 변수들 초기화
	initArrayToRandom(a, arrayWidth);
	initArrayToRandom(b, arrayWidth);

	//두 정방행렬의 곱셈연산(CPU에서)
	initArrayToZero(c, arrayWidth);
	squareMatrixMulWithCPU(c, a, b, arrayWidth);


	/*
	printArrayAllElement(a, arrayWidth);
	printf("\n");
	printArrayAllElement(b, arrayWidth);
	printf("\n");
	*/

	//printArrayAllElement(c, arrayWidth);
	//printf("\n");


	//두 정방행렬의 곱셈연산(GPU에서)
	initArrayToZero(c, arrayWidth);
	cudaError_t cudaStatus = squareMatrixMulWithGPU(c, a, b, arrayWidth);


	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "squareMatrixMulWithGPU failed!");
		return 1;
	}

	//결과확인
	//printArrayAllElement(c, arrayWidth);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

__global__ void squareMatrixMulKernel(int *c, int *a, int *b, int arrayWidth)
{
	float sum = 0;

	//여기서 threadIdx.x와 y는 행렬의 인덱스와 같다. 예시) 2x2행렬일때 00 01 10 11

	for (int i = 0; i < arrayWidth; ++i)
	{
		float Aelement = a[threadIdx.y * arrayWidth + i];
		float Belement = b[i*arrayWidth + threadIdx.x];
		sum += Aelement * Belement;
	}
	c[threadIdx.y * arrayWidth + threadIdx.x] = sum;
}

cudaError_t squareMatrixMulWithGPU(int *c, int *a, int *b, int arrayWidth)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	cudaError_t cudaStatus;


	dim3 dimGrid(1, 1);				// blocks per grid
	dim3 dimBlock(arrayWidth, arrayWidth);		// Threads per block


	//멀티 GPU 시스템 환경에서 실행할 GPU를 선택하는 코드. 
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Device(GPU)의 grid에 있는 Global Memory에 3개의 벡터를 위한 GPU버퍼를 할당한다.
	cudaStatus = cudaMalloc((void**)&dev_c, arrayWidth * arrayWidth * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, arrayWidth * arrayWidth * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, arrayWidth * arrayWidth * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// 호스트에 존재하는 버퍼를 Device(GPU)에 존재하는 GPU버퍼들로 복사한다.
	cudaStatus = cudaMemcpy(dev_a, a, arrayWidth * arrayWidth * sizeof(int), cudaMemcpyHostToDevice); //cudaMemcpy는 비동기전송으로 작동하며 HostToHost, HostToDevice, DeviceToHost, DeviceToDevice  4가지 타입이 가능함.
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, arrayWidth * arrayWidth * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEvent_t start, stop;
	float gapTime = 0;

	//이벤트 객체생성
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);	//시작시간 저장


	squareMatrixMulKernel << < dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, arrayWidth); // 왼쪽 2개의 변수는 쓰레드를 생성하는 조건, 오른쪽 4개의 변수들은 커널함수의 매개변수

	cudaEventRecord(stop, 0);	//끝난시간 저장

	cudaEventSynchronize(stop); 	//stop이벤트가 기록될 때 까지 여기서 멈춰있는다.


	cudaEventElapsedTime(&gapTime, start, stop); // 시작시간과 끝난 시간의 차를 계산하여 저장한다.

	//이벤트 객체제거
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	printf("연산시간 측정(GPU) : %f ms\n", gapTime);

	// 커널을 시작하는 동안 에러가 있었는지 확인한다.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "squareMatrixMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	// cudaDeviceSynchronize는 커널이 끝마칠 때 까지 기다린다. 그리고 그 실행동안에 발생했던 모든 오류를 반환한다.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching squareMatrixMulKernel!\n", cudaStatus);
		goto Error;
	}


	// GPU buffer에서 호스트 메모리로 결과벡터를 복사한다.
	cudaStatus = cudaMemcpy(c, dev_c, arrayWidth * arrayWidth * sizeof(int), cudaMemcpyDeviceToHost);
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

void squareMatrixMulWithCPU(int *c, int *a, int *b, int arrayWidth)
{
	clock_t startTime, endTime;
	double gapTime;

	startTime = clock(); //연산의 시작시간 체크


	for (int i = 0; i < arrayWidth; i++) {
		for (int j = 0; j < arrayWidth; j++) {
			int sum = 0;

			for (int k = 0; k < arrayWidth; k++) {
				int hos_a = a[i*arrayWidth + k];
				int hos_b = b[k*arrayWidth + j];
				//printf("%d * %d\n", i*arrayWidth + k, k*arrayWidth + j);
				sum += hos_a*hos_b;
			}
			c[i*arrayWidth + j] = sum;
		}
	}

	endTime = clock();	//연산의 끝난시간 체크

	gapTime = (double)endTime - startTime;

	printf("연산시간 측정(CPU) : %f ms\n", gapTime);
}

void initArrayToRandom(int *array, int arrayWidth)
{
	int arrayTotalCount = arrayWidth * arrayWidth;

	for (int i = 0; i < arrayTotalCount; i++)
		array[i] = rand() % 2; // 0 ~ 4-1 범위의 랜덤정수 생성.
}

void initArrayToZero(int *array, int arrayWidth)
{
	int arrayTotalCount = arrayWidth * arrayWidth;

	for (int i = 0; i < arrayTotalCount; i++)
		array[i] = 0;
}

void printArrayAllElement(int *array, int arrayWidth)
{
	int arrayTotalCount = arrayWidth * arrayWidth;

	for (int i = 0; i < arrayTotalCount; i++)
	{
		printf("%2d ", array[i]);
		if ((i + 1) % arrayWidth == 0)
			printf("\n");
	}
}
