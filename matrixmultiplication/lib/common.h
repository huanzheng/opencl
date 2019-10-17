// Common C includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

#define ROWMAJOR true

// =================================================================================================

// Repeat all kernels multiple times to get an average timing result
#define NUM_RUNS 4

// Squared matrices are tested within a certain range (e.g. 1024x1024, 2048x2048, 4096x4096)
#define MINSIZE (1024)
#define MAXSIZE (4*1024)
// OpenCL settings
#define MAX_NUM_DEVICES 16
#define MAX_DEVICE_NAME 1024
#define CURRENT_DEVICE 0

// =================================================================================================

// Timer structure
typedef struct {
    double t; // Time
    int long long kf; // KFlops
} profile_t;

// Number of timers
#define NUM_TIMERS 10
// Forward declarations of the timer functions
double timer(void);
double wtime(profile_t timer);
double gflops(profile_t timer);

// Other forward declarations
char* readKernelFile(const char* filename, long* _size);


// matrix related
float* mallocFloatMatrix(int m, int n);
void setFloatMatrixItem(float* data, int Mrow, int Ncol, int mrow, int ncol, float value, bool rowMajor);
float getFloatMatrixItem(float* data, int Mrow, int Ncol, int mrow, int ncol, bool rowMajor);
void printFloatMatrix(float* data, int Mrow, int Ncol, bool rowMajor);
void floatMatrixMultiple(float* a, float* b, float* c, int M, int N, int K, bool rowMajor);
