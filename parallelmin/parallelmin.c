#include <CL/cl.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>

double timer(void) {
    struct timeval Tvalue;
    struct timezone dummy;
    gettimeofday(&Tvalue, &dummy);
    double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    return etime;
    //return omp_get_wtime();
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


cl_uint getminimal(cl_uint *src_ptr, int size) {
	cl_uint ret = (cl_uint) -1;
	for (int i = 0; i < size; i++)
    	{
		ret = src_ptr[i] < ret ? src_ptr[i]:ret;
    	}
	return ret;
}

#define NLOOPS 500
int main() {

	cl_uint *src_ptr;
	unsigned int num_src_items = 4096*4096;
	src_ptr = (cl_uint*) malloc(num_src_items * sizeof(cl_uint));

	time_t ltime;
	time(&ltime);

	cl_uint a = (cl_uint)ltime;
	cl_uint b = (cl_uint)ltime;
	cl_uint min = (cl_uint) -1;

	for (int i = 0; i < num_src_items; i++) {
		src_ptr[i] = (cl_uint) (b = (a * (b & 65536)) + (b>>16));
		min = src_ptr[i] < min ? src_ptr[i]:min;
	}
	printf("min is %u\n",min);

	double startOfCPU = timer();
	cl_uint cpumini;
	for (int i = 0; i < NLOOPS; i++) {
		cpumini = getminimal(src_ptr, num_src_items);
	}
	double elaspedCPU = timer() - startOfCPU;
	printf("cpumini is %u, Avg min time Of CPU is %.6f\n",cpumini, elaspedCPU/NLOOPS);

#define MAX_NUM_DEVICES 1
#define MAX_DEVICE_NAME 1000
#define CURRENT_DEVICE 0
#define DEV 1
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_int err;
    	cl_platform_id platform = 0;
    	cl_device_id device = 0;
    	cl_device_id devices[MAX_NUM_DEVICES];
    	cl_uint numDevices = 0;
    	cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
	int dev = DEV;

    	cl_event event = NULL;
    	char deviceName[MAX_DEVICE_NAME];


    	// Configure the OpenCL environment
    	err = clGetPlatformIDs(1, &platform, NULL);
    	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    	device = devices[CURRENT_DEVICE];
    	props[1] = (cl_context_properties)platform;
    	context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    	queue = clCreateCommandQueue(context, device, 0, &err);
    	err = clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_DEVICE_NAME, deviceName, NULL);
    	checkError(err,__LINE__);
    	size_t maxwgs;
    	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwgs), &maxwgs, NULL);
    	cl_uint compute_units;
    	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
	printf("## %d devices, running on %d: '%s', MAX_WORK_GROUP_SIZE is %ld, MAX_CU %ld\n", numDevices, CURRENT_DEVICE, deviceName, maxwgs, compute_units);

	size_t global_work_size;
	size_t local_work_size;
	size_t num_groups;

	cl_uint ws = 64;
	global_work_size = compute_units * 7 * ws; //7 wavefronts per SIMD
	while((num_src_items/4) % global_work_size != 0)
		global_work_size += ws;
	local_work_size = ws;

	num_groups = global_work_size/local_work_size;

   	 // Read the kernel file from disk
    	long sizeHeader, sizeSource;
    	char* source = readKernelFile("./kernel.cl", &sizeSource);

    	// Compile the kernel file
    	program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    	checkError(err,__LINE__);
    	err = clBuildProgram(program, 1, &device, "", NULL, NULL); // if add compile options, then getbuildinfo will report error invalid device

    	// Check for compilation errors
    	size_t logSize;
    	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    	checkError(err,__LINE__);
    	char* messages = (char*)malloc((1+logSize)*sizeof(char));
    	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    	checkError(err,__LINE__);
    	messages[logSize] = '\0';
    	if (logSize > 10) { printf("## Compiler message: %s\n", messages); }
    	free(messages);


	cl_kernel minp;
	cl_kernel reduce;
	cl_mem src_buf;
	cl_mem dst_buf;
	cl_mem dbg_buf;

	cl_uint *dst_ptr;
	cl_uint *dbg_ptr;

	minp = clCreateKernel(program, "minp", NULL);
	reduce = clCreateKernel(program, "reduce",NULL);

	src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				num_src_items*sizeof(cl_uint), src_ptr, NULL);

	dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, num_groups * sizeof(cl_uint), NULL, NULL);
	dbg_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, global_work_size * sizeof(cl_uint), NULL,NULL);

	clSetKernelArg(minp, 0, sizeof(void*), (void*) &src_buf);
	clSetKernelArg(minp, 1, sizeof(void*), (void*) &dst_buf);
	clSetKernelArg(minp, 2, 1 * sizeof(cl_uint), (void*) NULL);
	clSetKernelArg(minp, 3, sizeof(void*), (void*) &dbg_buf);
	clSetKernelArg(minp, 4, sizeof(num_src_items), (void*) &num_src_items);
	clSetKernelArg(minp, 5, sizeof(dev), (void*) &dev);

	clSetKernelArg(reduce, 0, sizeof(void*), (void*) &src_buf);
	clSetKernelArg(reduce, 1, sizeof(void*), (void*) &dst_buf);

	double startTime = timer();
	cl_event ev;
	int nloops = NLOOPS;
	while(nloops--) {
		clEnqueueNDRangeKernel(queue, minp, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev);
		clEnqueueNDRangeKernel(queue, reduce, 1, NULL, &num_groups, NULL, 1, &ev, NULL);
	}
	clFinish(queue);

	double elasped = timer() - startTime;
	printf("Avg min time Of GPU is %.6f\n", elasped/NLOOPS);

	dst_ptr = (cl_uint*) clEnqueueMapBuffer(queue, dst_buf, CL_TRUE, CL_MAP_READ, 0, num_groups * sizeof(cl_uint), 0, NULL, NULL, NULL);
	dbg_ptr = (cl_uint *) clEnqueueMapBuffer(queue, dbg_buf, CL_TRUE, CL_MAP_READ, 0, global_work_size * sizeof(cl_uint), 0, NULL, NULL, NULL);

	if (dst_ptr[0] == min)
		printf("result correct\n");
	else
		printf("result INcorrect\n");
}
