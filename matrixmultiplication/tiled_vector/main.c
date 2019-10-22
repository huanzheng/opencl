#include "common.h"
#include <CL/cl.h>
#include "settings.h"

#define MS 1024
#define KS 1024
#define NS 1024

void testmatrixmultiple();
void matrixMultipleInCL(float *A, float *B, float *C, int M, int K, int N);
void checkmatrixmultiplewithprint(int m, int k, int n);
void matrixMultipleInCPU(float *A, float* B, float* C, int M, int K, int N, bool rowMajor);

int main() {
	double t = timer();
	//testmatrixmultiple();
	checkmatrixmultiplewithprint(32,32,32);

	int size = 1024;
	for (size = 1024; size <=MAXSIZE; size=size*2) {
		int m = size;
		int n = size;
		int k = size;

		float *A = mallocFloatMatrix(m,k);
		float *B = mallocFloatMatrix(k,n);
		float *C1 = mallocFloatMatrix(m,n);
		float *C = mallocFloatMatrix(m,n);

		srand(time(NULL));

		int i;

		for(i = 0; i < m*k; i++) {
			A[i] = (float) rand() / (float) RAND_MAX;
		}
		for(i = 0; i < k*n; i++) {
			B[i] = (float) rand() / (float) RAND_MAX;
		}

		//printFloatMatrix(A, m, k, ROWMAJOR);
		//printFloatMatrix(B, k, n, ROWMAJOR);

		//matrixMultipleInCPU(A,B,C1,m,k,n,ROWMAJOR);
		//printFloatMatrix(C1, m, n, ROWMAJOR);

		matrixMultipleInCL(A,B,C,m,k,n);
	}
	//printFloatMatrix(C, m, n, ROWMAJOR);

}

void checkmatrixmultiplewithprint(int m, int k, int n) {
	float *A = mallocFloatMatrix(m,k);
        float *B = mallocFloatMatrix(k,n);
        float *C1 = mallocFloatMatrix(m,n);
        float *C = mallocFloatMatrix(m,n);

        srand(time(NULL));
        int i;

        for(i = 0; i < m*k; i++) {
	        A[i] = (float) rand() / (float) RAND_MAX;
        }
        for(i = 0; i < k*n; i++) {
                B[i] = (float) rand() / (float) RAND_MAX;
        }

        printFloatMatrix(A, m, k, ROWMAJOR);
        printFloatMatrix(B, k, n, ROWMAJOR);

        matrixMultipleInCPU(A,B,C1,m,k,n,ROWMAJOR);
        printFloatMatrix(C1, m, n, ROWMAJOR);

        matrixMultipleInCL(A,B,C,m,k,n);
	printFloatMatrix(C, m, n, ROWMAJOR);
}

void matrixMultipleInCPU(float *A, float* B, float* C, int M, int K, int N, bool rowMajor) {

	double startTime=timer();
	for (int i = 0; i < NUM_RUNS; i++) {
		floatMatrixMultiple(A,B,C,M,K,N,rowMajor);
	}

	double endTime = timer();
        double avgTime = (endTime-startTime)/NUM_RUNS;
        printf("Matrix %dx%d * %dx%d -> %dx%d takes %.6f seconds to complete on CPU\n",M,K,K,N,M,N,avgTime);
}




void matrixMultipleInCL(float *A, float *B, float *C, int M, int K, int N) {
	cl_int err;
	cl_event event;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	
	cl_preparation(&context, &queue, &program, "./tiled_vector.cl");

	// Prepare OpenCL memory objects
    	cl_mem bufA    = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(*A), NULL, &err);
    	cl_mem bufB    = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(*B), NULL, &err);
    	cl_mem bufC    = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);
    	checkError(err,__LINE__);

    	// Copy matrices to the GPU (also C to erase the results of the previous run)
    	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(*A), A, 0, NULL, NULL);
    	err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    	err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    	checkError(err,__LINE__);

    	char kernelname[100];
    	sprintf(kernelname, "%s", "tiled_vector");
    	cl_kernel kernel1 = clCreateKernel(program, kernelname, &err);
    	checkError(err,__LINE__);

	err = clSetKernelArg(kernel1, 0, sizeof(int), (void*)&M);
        err = clSetKernelArg(kernel1, 1, sizeof(int), (void*)&N);
        err = clSetKernelArg(kernel1, 2, sizeof(int), (void*)&K);
        err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&bufA);
        err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB);
        err = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&bufC);

	checkError(err,__LINE__);
	const size_t local[2] = { TS/WIDTH, TS };
        const size_t global[2] = { (size_t)(M/WIDTH), (size_t)N };

	// Start the timed loop
    	double startTime = timer();
    	for (int r=0; r<NUM_RUNS; r++) {
        	// Run the kernel
        	err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global, local, 0, NULL, &event);


        	// Wait for calculations to be finished
        	checkError(err,__LINE__);
        	err = clWaitForEvents(1, &event);
    	}
	double endTime = timer();
	double avgTime = (endTime-startTime)/NUM_RUNS;
	printf("Matrix %dx%d * %dx%d -> %dx%d takes %.6f seconds to complete on GPU\n",M,K,K,N,M,N,avgTime);

    	// End the timed loop
    	//timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    	//timers[timerID].kf += ((long)K * (long)M * (long)N * 2) / 1000;

    	// Copy the output matrix C back to the CPU memory
    	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    	checkError(err,__LINE__);

    	// Free the memory objects
    	clReleaseMemObject(bufA);
    	clReleaseMemObject(bufB);
    	clReleaseMemObject(bufC);

    	// Clean-up OpenCL
    	clReleaseCommandQueue(queue);
    	clReleaseContext(context);
    	clReleaseProgram(program);
    	clReleaseKernel(kernel1);
}


void testmatrixmultiple() {
	float* a= mallocFloatMatrix(3,3);
	float* b = mallocFloatMatrix(3,3);
	float* c = mallocFloatMatrix(3,3);
	int i = 0;
	for (i = 0; i<9; i++) {
		*(a+i) = (float) rand() / (float) RAND_MAX;
		*(b+i) = (float) rand() / (float) RAND_MAX;
	}
	bool rowmajor = false;
	printFloatMatrix(a, 3, 3, rowmajor);
	printFloatMatrix(b, 3, 3, rowmajor);

	for (int j = 0; j < 100; j++)
	floatMatrixMultiple(a, b, c, 3, 3, 3, rowmajor);
	printFloatMatrix(c, 3,3, rowmajor);
}
