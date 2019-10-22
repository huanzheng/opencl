#include "common.h"
#include <CL/cl.h>

#define CL_INCLUDE_FILE "../lib/settings.h"

// Determine the location where to output the PTX code
#define CL_PTX_FILE "bin/myGEMM.cl.ptx"

// Define OpenCL compiler options, such as "-cl-nv-maxrregcount=127"
#define COMPILER_OPTIONS ""


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

//MxK * KxN = MxN
void floatMatrixMultiple(float* a, float* b, float* c, int M, int K, int N, bool rowMajor) {
	int m,n,k;

	for (m = 0; m < M; m++) {
		for (n = 0; n < N; n++) {
			float* cAddr;
			float value = 0;
			if (rowMajor) {
				cAddr = c + m*N + n;
			} else {
				cAddr = c + n*M + m;
			}

			for (k = 0; k < K; k++) {
				if (rowMajor) {
					value += (*(a + m*K + k)) * (*(b + k*N + n));
				} else {
					value += (*(a + k*M + m)) * (*(b + n*K + k));
				}
			}
			*cAddr = value;
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

//opencl related below
//build the cl program
void cl_preparation(cl_context* context, cl_command_queue* queue, cl_program* program, const char* kernelpath) {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_device_id devices[MAX_NUM_DEVICES];
    cl_uint numDevices = 0;
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};

    cl_event event = NULL;
    char deviceName[MAX_DEVICE_NAME];


    // Configure the OpenCL environment
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    device = devices[CURRENT_DEVICE];
    props[1] = (cl_context_properties)platform;
    *context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    *queue = clCreateCommandQueue(*context, device, 0, &err);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_DEVICE_NAME, deviceName, NULL);
    checkError(err,__LINE__);
    size_t maxwgs;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwgs), &maxwgs, NULL);
    printf("## %d devices, running on %d: '%s', MAX_WORK_GROUP_SIZE is %ld\n", numDevices, CURRENT_DEVICE, deviceName, maxwgs);

    // Read the kernel file from disk
    long sizeHeader, sizeSource;
    char* header = readKernelFile(CL_INCLUDE_FILE, &sizeHeader);
    char* source = readKernelFile(kernelpath, &sizeSource);
    long size = sizeHeader + sizeSource - 1;
    char* code = (char*)malloc(size*sizeof(char));
    for (int c=0; c<size; c++) { code[c] = '\0'; }
    strcat(code, header);
    strcat(code, source);
    const char* constCode = code;
    free(header);
    free(source);

    // Compile the kernel file
    *program = clCreateProgramWithSource(*context, 1, &constCode, NULL, &err);
    checkError(err,__LINE__);
    err = clBuildProgram(*program, 0, NULL, COMPILER_OPTIONS, NULL, NULL);

    // Check for compilation errors
    size_t logSize;
    err = clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkError(err,__LINE__);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    err = clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    checkError(err,__LINE__);
    messages[logSize] = '\0';
    if (logSize > 10) { printf("## Compiler message: %s\n", messages); }
    free(messages);

    /*
    // Retrieve the PTX code from the OpenCL compiler and output it to disk
    size_t binSize;
    err = clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, NULL);
    checkError(err,__LINE__);
    unsigned char *bin = (unsigned char *)malloc(binSize);
    err = clGetProgramInfo(*program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
    checkError(err,__LINE__);
    FILE* file = fopen(CL_PTX_FILE, "wb");
    fwrite(bin, sizeof(char), binSize, file);
    fclose(file);
    free(bin);
	*/
}

// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
    if (error != CL_SUCCESS) {
        switch (error) {
            case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
            case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
            case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
            case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
            case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
            case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
            case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
            case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
            case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
            case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
            case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
            case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
            case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
            case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
            case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
            case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
            case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
            case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
            case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
            case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
            case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
            case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
            case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
            case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
            case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
            case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
            case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
            case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
            case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
            case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
            case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
            case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
            case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
            case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
            case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
            case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
            case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
            case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
            case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
            case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
            case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
            case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
            case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
            case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
            case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
            case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
            case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
            case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
            case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
            case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
            case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
            case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
            case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
            case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
            case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
            case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
            case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
            case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
            case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
            case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
            default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
        }
        exit(1);
    }
}

