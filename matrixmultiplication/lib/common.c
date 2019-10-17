#include "common.h"

float* mallocFloatMatrix(int m, int n) {
	float* ret = (float*) malloc(m*n*sizeof(float));
	return ret;
}

static float* getFloatMatrixItemAddr(float* data, int Mrow, int Ncol, int mrow, int ncol, bool rowMajor) {
	float *ret = 0;
        if(!rowMajor) {
                ret = data + ncol*Mrow + mrow;
        } else {
                ret = data + mrow*Ncol + ncol;
        }
	return ret;
}

void setFloatMatrixItem(float* data, int Mrow, int Ncol, int mrow, int ncol, float value, bool rowMajor) {
	float* addr = getFloatMatrixItemAddr(data, Mrow, Ncol, mrow, ncol, rowMajor);
	*addr = value;
}

float getFloatMatrixItem(float* data, int Mrow, int Ncol, int mrow, int ncol, bool rowMajor) {
	float* addr = getFloatMatrixItemAddr(data, Mrow, Ncol, mrow, ncol, rowMajor);
	return *addr;
}

void printFloatMatrix(float* data, int Mrow, int Ncol, bool rowMajor) {
	int row,col;
	for (row = 0; row < Mrow; row++) {
		for (col = 0; col < Ncol; col++) {
			printf("%.3f ", getFloatMatrixItem(data, Mrow, Ncol, row, col, rowMajor));
		}
		printf("\n");
	}
	printf("\n");
}

//MxN * NxK = MxK
void floatMatrixMultiple(float* a, float* b, float* c, int M, int N, int K, bool rowMajor) {
	int m,n,k;

	for (m = 0; m < M; m++) {
		for (k = 0; k < K; k++) {
			float* cAddr;
			if (rowMajor) {
				cAddr = c + m*K + k;
			} else {
				cAddr = c + k*M + m;
			}

			for (n = 0; n < N; n++) {
				if (rowMajor) {
					*cAddr += (*(a + m*N + n)) * (*(b + n*K + k));
				} else {
					*cAddr += (*(a + n*M + m)) * (*(b + k*N + n));
				}
			}
		}
	}
	return;
}

// Timer function: Measure the current time
double timer(void) {
    struct timeval Tvalue;
    struct timezone dummy;
    gettimeofday(&Tvalue, &dummy);
    double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    return etime;
    //return omp_get_wtime();
}

// Timer function: Get the execution time
double wtime(profile_t timer) {
    return (timer.t);
}

// Timer function: Get the GFLOPS number
double gflops(profile_t timer) {
    return ((double)timer.kf/(1000.0*1000.0)) / (timer.t);
}

char* readKernelFile(const char* filename, long* _size) {

    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("-- Error opening file %s\n", filename);
        exit(1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *)malloc((size+1)*sizeof(char));
    fread(source, 1, size*sizeof(char), file);
    source[size] = '\0';
    fclose(file);

    // Save the size and return the source string
    *_size = (size+1);
    return source;
}

