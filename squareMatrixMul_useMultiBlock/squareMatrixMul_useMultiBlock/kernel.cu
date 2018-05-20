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
	int *a;
	int *b;
	int *c;

	int arrayWidth;

	//이 함수가 메인함수에서 작동해야 같은 수가 안나오게됨.
	srand(time(NULL));

	while (1)
	{
		printf("행렬너비 : ");
		scanf("%d", &arrayWidth);

		a = (int*)malloc(sizeof(int)*arrayWidth*arrayWidth);
		b = (int*)malloc(sizeof(int)*arrayWidth*arrayWidth);
		c = (int*)malloc(sizeof(int)*arrayWidth*arrayWidth);


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

		free(a);
		free(b);
		free(c);
	}


	return 0;
}

__global__ void squareMatrixMulKernel(int *c, int *a, int *b, int arrayWidth)
{
	float sum = 0;

	//행렬에서 계산하려고 하는 위치의 인덱스 이것은 공식화 된것이므로 외우진 말자.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;


	//블록당 쓰레드가 4x4이고
	//블록의 개수가 1x1이면
	//printf("%d, %d / %d, %d / %d, %d\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
	// 4, 4, 0, 0, x, y 이렇게 앞에 4개의 숫자는 고정된 것을 볼 수 있었다.
	//blockDim : 블록 안쪽에 포함된 쓰레드가 어떤 ㅁxㅁ 차원으로 되어있는지.
	//blockIdx : 블록의 인덱스
	//threadIdx : 쓰레드의 인덱스

	for (int i = 0; i < arrayWidth; ++i)
	{
		float Aelement = a[row * arrayWidth + i];
		float Belement = b[i*arrayWidth + col];
		sum += Aelement * Belement;
	}
	c[row * arrayWidth + col] = sum;
}

cudaError_t squareMatrixMulWithGPU(int *c, int *a, int *b, int arrayWidth)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	cudaError_t cudaStatus;


	//최적의 블록 하나당 쓰레드의 수가 16x16이라고 하자.
	//그러면 16보다 작으면서 arrayWidth의 공약수인 수를 찾아서 그 수를 블록 하나당 쓰레드의 수를 지정해줘야 한다. 
	int threadWidth = 0;

	for (int i = 16; i > 0; --i)
	{
		if (arrayWidth % i == 0)
		{
			threadWidth = i;
			break;
		}
	}

	//여기서 dimBlock의 크기는 입력받은 배열의 너비에 따라 달라져야 한다. 안그러면 연산이 틀어져 잘못된 결과를 받아볼 수 있을 것이다. 
	dim3 dimBlock(threadWidth, threadWidth);												// 블록 하나 당 쓰레드 수 
	dim3 dimGrid(arrayWidth / dimBlock.x, arrayWidth / dimBlock.y);		// 생성할 블록의 개수

																		//예시) 너비가 12인 행렬 즉 12*12행렬이 존재한다 치자. 그런데 우리는 블록 하나당 4*4의 쓰레드를 가지게 하였으므로 한번에 병렬처리로 연산해버리려면 블록은 3*3개의 블록이 필요하다.
	printf("블록당 쓰레드 수 : %d x %d, 블록의 수 : %d x %d\n\n", threadWidth, threadWidth, arrayWidth / dimBlock.x, arrayWidth / dimBlock.y);


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

	endTime = clock();//연산의 끝난시간 체크

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
