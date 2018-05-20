//GPU�������� ó���غ��� ����
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h> //clock(), time_tŸ���� ����

__global__ void squareMatrixMulKernel(int *c, int *a, int *b, int arrayWidth);	//�ٷ� ���� ������ �Լ� �ȿ��� ������ ������ �Բ� ȣ��Ǵ� �Լ�, host���� ȣ�Ⱑ���ϸ� Device���� ����Ǵ� Ŀ�� �Լ�
cudaError_t squareMatrixMulWithGPU(int *c, int *a, int *b, int arrayWidth);		// �� ��������� ���������� GPU���� �����ϴ� �Լ�
void squareMatrixMulWithCPU(int *c, int *a, int *b, int arrayWidth);			// �� ��������� ���������� CPU���� �����ϴ� �Լ� 
void initArrayToRandom(int *array, int arrayWidth);								// ������ ���� ����� �ʱ�ȭ �ϴ� �Լ� 
void initArrayToZero(int *array, int arrayWidth);								// 0���� ����� �ʱ�ȭ �ϴ� �Լ�
void printArrayAllElement(int *array, int arrayWidth);							// ����� ��� ���Ҹ� ����ϴ� �Լ�

int main()
{
	const int arrayWidth = 16;	//��� �Ѱ����� �̿��� �����Ͽ����Ƿ� �� ��Ͽ� �ִ� 1024���� �����常 ��밡���ϰ� ���� 32�� �ִ� ���ġ�̴�. �̸� �Ѿ�� GPU���� ������ ���� ���̴�.

								//�Ʒ��� �ּ�ó���� ���� ������ε� ���� ����
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

	//�� �Լ��� �����Լ����� �۵��ؾ� ���� ���� �ȳ����Ե�.
	srand(time(NULL));

	//������ �����ϱ��� ������ �ʱ�ȭ
	initArrayToRandom(a, arrayWidth);
	initArrayToRandom(b, arrayWidth);

	//�� ��������� ��������(CPU����)
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


	//�� ��������� ��������(GPU����)
	initArrayToZero(c, arrayWidth);
	cudaError_t cudaStatus = squareMatrixMulWithGPU(c, a, b, arrayWidth);


	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "squareMatrixMulWithGPU failed!");
		return 1;
	}

	//���Ȯ��
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

	//���⼭ threadIdx.x�� y�� ����� �ε����� ����. ����) 2x2����϶� 00 01 10 11

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


	dim3 dimGrid(1, 1);							// blocks per grid
	dim3 dimBlock(arrayWidth, arrayWidth);		// Threads per block


												//��Ƽ GPU �ý��� ȯ�濡�� ������ GPU�� �����ϴ� �ڵ�. 
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Device(GPU)�� grid�� �ִ� Global Memory�� 3���� ���͸� ���� GPU���۸� �Ҵ��Ѵ�.
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

	// ȣ��Ʈ�� �����ϴ� ���۸� Device(GPU)�� �����ϴ� GPU���۵�� �����Ѵ�.
	cudaStatus = cudaMemcpy(dev_a, a, arrayWidth * arrayWidth * sizeof(int), cudaMemcpyHostToDevice); //cudaMemcpy�� �񵿱��������� �۵��ϸ� HostToHost, HostToDevice, DeviceToHost, DeviceToDevice  4���� Ÿ���� ������.
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

	//�̺�Ʈ ��ü����
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);		 //���۽ð� ����


	squareMatrixMulKernel << < dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, arrayWidth); // ���� 2���� ������ �����带 �����ϴ� ����, ������ 4���� �������� Ŀ���Լ��� �Ű�����

	cudaEventRecord(stop, 0);		//�����ð� ����

	cudaEventSynchronize(stop); 	//stop�̺�Ʈ�� ��ϵ� �� ���� ���⼭ �����ִ´�.


	cudaEventElapsedTime(&gapTime, start, stop); // ���۽ð��� ���� �ð��� ���� ����Ͽ� �����Ѵ�.

												 //�̺�Ʈ ��ü����
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	printf("����ð� ����(GPU) : %f ms\n", gapTime);

	// Ŀ���� �����ϴ� ���� ������ �־����� Ȯ���Ѵ�.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "squareMatrixMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	// cudaDeviceSynchronize�� Ŀ���� ����ĥ �� ���� ��ٸ���. �׸��� �� ���ൿ�ȿ� �߻��ߴ� ��� ������ ��ȯ�Ѵ�.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching squareMatrixMulKernel!\n", cudaStatus);
		goto Error;
	}


	// GPU buffer���� ȣ��Ʈ �޸𸮷� ������͸� �����Ѵ�.
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

	startTime = clock(); //������ ���۽ð� üũ


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

	endTime = clock();	//������ �����ð� üũ

	gapTime = (double)endTime - startTime; // �����Ŭ����ƽ�� / �ʴ�Ŭ����

	printf("����ð� ����(CPU) : %f ms\n", gapTime);
}

void initArrayToRandom(int *array, int arrayWidth)
{
	int arrayTotalCount = arrayWidth * arrayWidth;

	for (int i = 0; i < arrayTotalCount; i++)
		array[i] = rand() % 2; // 0 ~ 4-1 ������ �������� ����.
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