#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/time.h>

#ifndef FPGA_DEVICE
#include "lu259_cl.h"
#endif

#define BLOCK_SIZE 16
#define ITER 1 
// Note: Iterations not implemented yet


int load_file_to_memory(const char *filename, char **result) { 

	int size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) 
	{ 
		*result = NULL;
		return -1; // -1 means file opening fail 
	} 
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f)) 
	{ 
		free(*result);
		return -2; // -2 means file reading fail 
	} 
	fclose(f);
	(*result)[size] = 0;

	return size;
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	int err = 0;
	int passed = 0;
	// timer structs
    double elapsed = 0;
	srand(time(NULL));
	
	int N = 64;
	
	char dir[100] = "./data";

	if (argc>1)
		N = atoi(argv[1]);

	if (argc>2)
		strcpy(dir, argv[2]);
		
	
	
	// Initialize A and b
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			double r = (double) rand();
			if(r > RAND_MAX/2)
				A[i*N+j] = A0[i*N+j] = -(r-RAND_MAX/2)/(RAND_MAX/2);
			else
				A[i*N+j] = A0[i*N+j] = r/(RAND_MAX/2);
		}
		double r = (double) rand();
		if(r > RAND_MAX/2)
			b[i] = b0[i] = -(r-RAND_MAX/2)/(RAND_MAX/2);
		else
			b[i] = b0[i] = r/(RAND_MAX/2);
	}
	
	// Initialize L matrix, x,y vectors
	// Added to ensure initial values are 0
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			L[i*N+j] = 0;
		}
		y[i] = 0;
		x[i] = 0;
	}
	
	
	// Allocate matrices and vectors
	double *A = (double *) malloc(N*N*sizeof(double));
	double *A0 = (double *) malloc(N*N*sizeof(double));
	double *b = (double *) malloc(N*sizeof(double));
	double *b0 = (double *) malloc(N*sizeof(double)); // ADDED; original b matrix before permutations
	double *L = (double *) malloc(N*N*sizeof(double));
	double *x = (double *) malloc(N*sizeof(double));
	double *y = (double *) malloc(N*sizeof(double));
	
	
	// 1. allocate host memory for matrices A and B
	int width_A, width_A0, width_L, height_A, height_A0, height_L, height_b, height_b0, height_x, height_y;
	width_A = width_A0 = width_L = height_A = height_A0 = height_L = height_b = height_b0 = height_x = height_y = N;
	
	unsigned int size_A = width_A * height_A;
	unsigned int size_A0 = width_A0 * height_A0;
	unsigned int size_L = width_L * height_L;
	unsigned int size_b = height_b;
	unsigned int size_b0 = height_b0;
	unsigned int size_x = height_x;
	unsigned int size_y = height_y;
	unsigned int mem_size_A = sizeof(double) * size_A;
	unsigned int mem_size_A0 = sizeof(double) * size_A0;
	unsigned int mem_size_L = sizeof(double) * size_L;
	unsigned int mem_size_b = sizeof(double) * size_b;
	unsigned int mem_size_b0 = sizeof(double) * size_b0;
	unsigned int mem_size_x = sizeof(double) * size_x;
	unsigned int mem_size_y = sizeof(double) * size_y;
	
	// Host pointers
	double* h_A = A;
	//double* h_A0 = A0;
	double* h_L = L;
	double* h_b = b;
	//double* h_b0 = b0;
	double* h_x = x;
	double* h_y = y;
	
	
	// 5. Initialize OpenCL
     
	cl_command_queue clCommandQue;
	cl_program program;
	cl_kernel clKernel;

	size_t dataBytes;
	size_t kernelLength;
	cl_int status;
	
	
	/*****************************************/
	/* Initialize OpenCL */
	/*****************************************/

	// Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

	//printf("Found %d platforms support OpenCL, return code %d.\n", numPlatforms, status);
 
    // Allocate enough space for each platform
    cl_platform_id *platforms = NULL;

    platforms = (cl_platform_id*)malloc( numPlatforms*sizeof(cl_platform_id));
 
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS)
		printf("clGetPlatformIDs error(%d)\n", status);
	
	// Retrieve the number of devices
    cl_uint numDevices = 0;
#ifndef FPGA_DEVICE
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
#else
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
#endif
	printf("Found %d devices support OpenCL.\n", numDevices);

    // Allocate enough space for each device
    cl_device_id *devices = (cl_device_id*)malloc( numDevices*sizeof(cl_device_id));

    // Fill in the devices 
#ifndef FPGA_DEVICE
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
#else
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
#endif

	if (status != CL_SUCCESS)
		printf("clGetDeviceIDs error(%d)\n", status);

    // Create a context and associate it with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if (status != CL_SUCCESS)
		printf("clCreateContext error(%d)\n", status);

	// OpenCL device memory for matrices
	cl_mem d_A;
	//cl_mem d_A0;
	cl_mem d_L;
	cl_mem d_b;
	//cl_mem d_b0;
	cl_mem d_x;
	cl_mem d_y;
	
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(context, devices[0], 0, &status);

	if (status != CL_SUCCESS)
		printf("clCreateCommandQueue error(%d)\n", status);

	// Setup device memory
	d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_x, NULL, &status);
	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &status);
	//d_A0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A0, h_A0, &status);
	d_L = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_L, h_L, &status);
	d_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_b, h_b, &status);
	//d_b0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_b0, h_b0, &status);
	d_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_y, h_y, &status);

#ifndef FPGA_DEVICE
	// WE CAN'T USE THIS UNLESS WE MAKE A HEADER FILE WITH A GIANT STRING OF THE KERNEL PROGRAM
	// Create a program with source code
    program = clCreateProgramWithSource(context, 1, 
        (const char**)&lu259_cl, NULL, &status);
	if (status != 0)
		printf("clCreateProgramWithSource error(%d)\n", status);

    // Build (compile) the program for the device
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	
#else
	// Load binary from disk
	unsigned char *kernelbinary;
	char *xclbin = argv[3];
	printf("loading %s\n", xclbin);
	int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (n_i < 0) {
		printf("ERROR: failed to load kernel from xclbin: %s\n", xclbin);
		return -1;
	}
	size_t n_bit = n_i;

	// Create the compute program from offline
	program = clCreateProgramWithBinary(context, 1, &devices[0], &n_bit,
			(const unsigned char **) &kernelbinary, NULL, &status);
	if ((!program) || (status != CL_SUCCESS)) {
		printf("Error: Failed to create compute program from binary %d!\n", status);
		return -1;
	}

	// Build the program executable
	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#endif

	if (status != 0) {
		char errmsg[2048];
		size_t sizemsg = 0;

		status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 2048*sizeof(char), errmsg, &sizemsg);

		printf("clBuildProgram error(%d)\n", status);
		printf("Compilation messages: \n %s", errmsg);
	}

	clKernel = clCreateKernel(program, "LUFact", &status);
	if (status != CL_SUCCESS)
		printf("clCreateKernel error(%d)\n", status);

	// 7. Launch OpenCL kernel
	//size_t localWorkSize[2], globalWorkSize[2];
	size_t localWorkSize, globalWorkSize;
	
	int width_matrix = width_A;
	int height_vector = height_x;
	
	status  = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_x);
	status |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_A);
	status |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_L);
	status |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&d_b);
	status |= clSetKernelArg(clKernel, 4, sizeof(cl_mem), (void *)&d_y);
	status |= clSetKernelArg(clKernel, 5, sizeof(int), (void *)&width_matrix);
	status |= clSetKernelArg(clKernel, 6, sizeof(int), (void *)&height_vector);
	
	if (status != CL_SUCCESS)
		printf("clSetKernelArg error(%d)\n", status);
		
	
	//localWorkSize[0] = BLOCK_SIZE;
	//localWorkSize[1] = BLOCK_SIZE;
	//globalWorkSize[0] = width_A;
	//globalWorkSize[1] = height_A;
	localWorkSize = width_A; 
	globalWorkSize = width_A; // One work group and N work-items

    // start timer
	clock_t start = clock();

    status = clEnqueueWriteBuffer(clCommandQue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0, NULL, NULL);
	//status = clEnqueueWriteBuffer(clCommandQue, d_A0, CL_FALSE, 0, mem_size_A0, h_A0, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(clCommandQue, d_L, CL_FALSE, 0, mem_size_L, h_L, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(clCommandQue, d_b, CL_FALSE, 0, mem_size_b, h_b, 0, NULL, NULL);
	//status = clEnqueueWriteBuffer(clCommandQue, d_b0, CL_FALSE, 0, mem_size_b0, h_b0, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(clCommandQue, d_y, CL_FALSE, 0, mem_size_y, h_y, 0, NULL, NULL);
	
	status = clEnqueueNDRangeKernel(clCommandQue, 
			clKernel, 1, NULL, globalWorkSize, 
			localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		printf("clEnqueueNDRangeKernel error(%d)\n", status);

	// 8. Retrieve result from device
	status = clEnqueueReadBuffer(clCommandQue, d_x, CL_TRUE, 0, mem_size_x, h_x, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		printf("clEnqueueReadBuffer error(%d)\n", status);
	
	// stop timer
	clock_t end = clock();
	elapsed += ((double)(end-start)) / CLOCKS_PER_SEC;
	
	// Check result
	double error = 0;
	for(int i = 0; i < N; i++)
	{
		double b_res = 0;
		for(int j = 0; j < N; j++)
			b_res += A0[i*N+j] * x[j];
		error += b_res > b0[i] ? b_res-b0[i] : b0[i]-b_res;
	}
		
	double epsilonPerRow = 0.01;
	if(error < N*epsilonPerRow)
		passed++;
		
	printf("%d of %d tests passed\n", passed, ITER);
	printf("Average time: %.2f seconds\n", elapsed/ITER);
	
	// 10. clean up memory
	free(h_A);
	free(h_A0);
	free(h_L);
	free(h_b);
	free(h_b0);
	free(h_x);
	free(h_y);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_L);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_y);

	free(devices);
	clReleaseContext(context);
	clReleaseKernel(clKernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(clCommandQue);
}
	
	
	
	
