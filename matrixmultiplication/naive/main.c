#include "common.h"
#include <CL/cl.h>

#define M 32
#define K 32
#define N 32

void testmatrixmultiple();

int main() {
	double t = timer();
	float *a = mallocFloatMatrix(M,N);
	printFloatMatrix(a, M, N, ROWMAJOR);

	cl_context context;
	cl_command_queue queue;
	cl_program program;
	
	cl_preparation(&context, &queue, &program, "./naive.cl");
}


void testmatrixmultiple() {
	float* a= mallocFloatMatrix(2,3);
	float* b = mallocFloatMatrix(3,2);
	float* c = mallocFloatMatrix(2,2);
	int i = 0;
	for (i = 0; i<6; i++) {
		*(a+i) = i;
		*(b+i) = i;
	}
	bool rowmajor = false;
	printFloatMatrix(a, 2, 3, rowmajor);
	printFloatMatrix(b, 3, 2, rowmajor);

	floatMatrixMultiple(a, b, c, 2, 3, 2, rowmajor);
	printFloatMatrix(c, 2,2, rowmajor);
}
