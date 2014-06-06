#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <time.h>

#ifndef FPGA_DEVICE
#include "lu259_cl.h"
#endif

#define BLOCK_SIZE 128
#define BLOCK_SIZE_SUB 128
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

void show_matrix(float * matrix, char * fmt, int N)
{
	int i, j;
	if (!fmt) fmt = "%8.4g";
	for (i = 0; i < N; i++)
	{
		printf(i ? "      " : " [ ");
		for (j = 0; j < N; j++)
		{
			printf(fmt, matrix[i*N+j]);
			printf(j < N - 1 ? "  " : i == N - 1 ? " ]\n" : "\n");
		}
	}
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	int err = 0;
	int passed = 0;
	// timer structs
	struct timeval t1, t2, tr;
    double elapsed = 0;
	srand(time(NULL));
	
	int N = 4;
	
	char dir[100] = "./data";

	if (argc>1)
		N = atoi(argv[1]);

	//if (argc>2)
	//	strcpy(dir, argv[2]);
	
	
	
	// Allocate matrices and vectors
	float *A = (float *) malloc(N*N*sizeof(float));
	float *A0 = (float *) malloc(N*N*sizeof(float));
	float *b = (float *) malloc(N*sizeof(float));
	float *b0 = (float *) malloc(N*sizeof(float)); // ADDED; original b matrix before permutations
	float *L = (float *) malloc(N*N*sizeof(float));
	float *x = (float *) malloc(N*sizeof(float));
	float *y = (float *) malloc(N*sizeof(float));
	float *Acurr = (float *) malloc(N*sizeof(float));
	float *denom = (float *) malloc(sizeof(float));
	float *nextDenom = (float *) malloc(sizeof(float));
	float *yPart = (float *) malloc((N/BLOCK_SIZE_SUB)*sizeof(float));
	float *xPart = (float *) malloc((N/BLOCK_SIZE_SUB)*sizeof(float));
	
	int i, j;
	// Initialize A and b
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			double r = (double) /*(-10+(rand() % 21));*/rand();
			//A[i*N+j] = A0[i*N+j] = r;
			if(r > RAND_MAX/2)
				A[i*N+j] = A0[i*N+j] = -(r-RAND_MAX/2)/(RAND_MAX/2);
			else
				A[i*N+j] = A0[i*N+j] = r/(RAND_MAX/2);
		}
		double r = (double) /*(-10+(rand() % 21));*/rand();
		//b[i] = b0[i] = r;
		if(r > RAND_MAX/2)
			b[i] = b0[i] = -(r-RAND_MAX/2)/(RAND_MAX/2);
		else
			b[i] = b0[i] = r/(RAND_MAX/2);
	}
	
	// Initialize L matrix, x,y vectors
	// Added to ensure initial values are 0
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			L[i*N+j] = 0;
		}
		y[i] = 0;
		x[i] = 0;
		Acurr[i] = 0;
	}
	
	for (i = 0; i < N/BLOCK_SIZE; i++)
	{
		yPart[i] = 0;
		xPart[i] = 0;
	}
	
	// TEST A AND b MANUAL GENERATION
	/*
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				if (i == j)
					A[i*N+j] = A0[i*N+j] = 1;
				else
					A[i*N+j] = A0[i*N+j] = 0;
			}
			b[i] = b0[i] = (float) i/(10.0);
		}
	*/	
		// END GENERATION
	
	//show_matrix(A,0,N);
	
	
	// 1. allocate host memory for matrices A and B
	int width_A, width_A0, width_L, height_A, height_A0, height_L, height_b, height_b0, height_x, height_y, width_Acurr;
	width_A = width_A0 = width_L = height_A = height_A0 = height_L = height_b = height_b0 = height_x = height_y = width_Acurr = N;
	
	unsigned int size_A = width_A * height_A;
	unsigned int size_A0 = width_A0 * height_A0;
	unsigned int size_L = width_L * height_L;
	unsigned int size_b = height_b;
	unsigned int size_b0 = height_b0;
	unsigned int size_x = height_x;
	unsigned int size_y = height_y;
	unsigned int size_Acurr = width_Acurr;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int mem_size_A0 = sizeof(float) * size_A0;
	unsigned int mem_size_L = sizeof(float) * size_L;
	unsigned int mem_size_b = sizeof(float) * size_b;
	unsigned int mem_size_b0 = sizeof(float) * size_b0;
	unsigned int mem_size_x = sizeof(float) * size_x;
	unsigned int mem_size_y = sizeof(float) * size_y;
	unsigned int mem_size_Acurr = sizeof(float) * size_Acurr;
	unsigned int mem_size_denom = sizeof(float);
	unsigned int mem_size_nextDenom = sizeof(float);
	unsigned int mem_size_yPart = sizeof(float) * (N/BLOCK_SIZE_SUB);
	unsigned int mem_size_xPart = sizeof(float) * (N/BLOCK_SIZE_SUB);
	
	// Host pointers
	float* h_A = A;
	float* h_L = L;
	float* h_b = b;
	float* h_x = x;
	float* h_y = y;
	float* h_Acurr = Acurr;
	float* h_denom = denom;
	float* h_nextDenom = nextDenom;
	float* h_yPart = yPart;
	float* h_xPart = xPart;
	
	
	// 5. Initialize OpenCL
     
	cl_command_queue clCommandQue;
	cl_program program;
	cl_kernel clKernel;
	cl_kernel clKernelFSub; // Kernel for forward substitution
	cl_kernel clKernelBSub; // Kernel for backward substitution

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

		
	// GET MAX DEVICE LOCAL MEMORY SIZE	
	//cl_ulong mem_size;
	//clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    //printf("CL_DEVICE_LOCAL_MEM_SIZE: %d KB\n", (unsigned int)(mem_size / 1024));
	
	// GET MAX NUMBER OF WORK ITEMS PER DIMENSION
	//size_t workitem_size[3];
	//cl_int ret = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
	//printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: %d / %d / %d\n", workitem_size[0], workitem_size[1], workitem_size[2]);
	
    // Create a context and associate it with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if (status != CL_SUCCESS)
		printf("clCreateContext error(%d)\n", status);

	// OpenCL device memory for matrices
	cl_mem d_A;
	cl_mem d_L;
	cl_mem d_b;
	cl_mem d_x;
	cl_mem d_y;
	cl_mem d_Acurr;
	cl_mem d_denom;
	cl_mem d_nextDenom;
	cl_mem d_yPart;
	cl_mem d_xPart;
	
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(context, devices[0], 0, &status);

	if (status != CL_SUCCESS)
		printf("clCreateCommandQueue error(%d)\n", status);

	// Setup device memory
	d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_x, h_x, &status);
	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &status);
	d_L = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_L, h_L, &status);
	d_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_b, h_b, &status);
	d_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_y, h_y, &status);
	d_Acurr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_Acurr, h_Acurr, &status);
	d_denom = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_denom, h_denom, &status);
	d_nextDenom = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_nextDenom, NULL, &status);
	d_yPart = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_yPart, NULL, &status);
	d_xPart = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_xPart, NULL, &status);

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
	char *xclbin = argv[2];
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
	
	status = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_A);
	status |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_denom);
	status |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_nextDenom);
	status |= clSetKernelArg(clKernel, 4, sizeof(int), (void *)&N);
	if (status != CL_SUCCESS)
			printf("clSetKernelArg error(%d)\n", status);

			
	clKernelFSub = clCreateKernel(program, "fSub", &status);
	if (status != CL_SUCCESS)
		printf("clCreateKernel error(%d)\n", status);

	status = clSetKernelArg(clKernelFSub, 0, sizeof(cl_mem), (void *)&d_A);
	status |= clSetKernelArg(clKernelFSub, 1, sizeof(cl_mem), (void *)&d_y);
	status |= clSetKernelArg(clKernelFSub, 2, sizeof(cl_mem), (void *)&d_yPart);
	status |= clSetKernelArg(clKernelFSub, 4, sizeof(int), (void *)&N);
	if (status != CL_SUCCESS)
		printf("clSetKernelArg error(%d)\n", status);
		
		
	clKernelBSub = clCreateKernel(program, "bSub", &status);
	if (status != CL_SUCCESS)
		printf("clCreateKernel error(%d)\n", status);
				
	status = clSetKernelArg(clKernelBSub, 0, sizeof(cl_mem), (void *)&d_A);
	status |= clSetKernelArg(clKernelBSub, 1, sizeof(cl_mem), (void *)&d_x);
	status |= clSetKernelArg(clKernelBSub, 2, sizeof(cl_mem), (void *)&d_xPart);
	status |= clSetKernelArg(clKernelBSub, 4, sizeof(int), (void *)&N);	
	if (status != CL_SUCCESS)
		printf("clSetKernelArg error(%d)\n", status);
		
	
	// 7. Launch OpenCL kernel
	
	// start timer
	clock_t start = clock();
	gettimeofday(&t1, NULL);
	status = clEnqueueWriteBuffer(clCommandQue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0, NULL, NULL);
	
			
	size_t localWorkSize[1], globalWorkSize[1];
	// Ready for pivoting
	*denom = A[0];
	
	int n, z;
	for (n = 0; n < N-1; n++)
	{
		//printf("denom: %f\n", *denom);
		if(*denom == 0.0)
		{
			// PARTIAL PIVOTING FOR ROWS OF [A b]		
			status = clEnqueueReadBuffer(clCommandQue, d_A, CL_TRUE, 0, mem_size_A, h_A, 0, NULL, NULL);
			if (status != CL_SUCCESS)
				printf("clEnqueueReadBuffer error(%d)\n", status);
			int max_j = n;
			// Search for the maximum value in the column, starting from the current row
			for (j = n; j < N; j++)
			{
				if (fabs(A[j*N+n]) > fabs(A[max_j*N+n])) 
				{
					max_j = j;
				}
			}
						
			// Swap rows for partial pivoting
			if (max_j != n)
			{
				for (z = 0; z < N; z++) 
				{ 
					float temp = A[n*N+z];
					A[n*N+z] = A[max_j*N+z];
					A[max_j*N+z] = temp;	
				}
				float temp_b = b[n];
				b[n] = b[max_j];
				b[max_j] = temp_b;
			}		
			*denom = A[n*N+n]; // Update denom with new value
			//printf("denom in pivot update: %f\n", *denom);
			status = clEnqueueWriteBuffer(clCommandQue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0, NULL, NULL);
		}
		//show_matrix(A,0,N);
		
		//status  = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_x);
		//status |= clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_A);
		status |= clSetKernelArg(clKernel, 3, sizeof(int), (void *)&n);
		//status |= clSetKernelArg(clKernel, 2, sizeof(int), (void *)&N);
		//status |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_L);
		//status |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&d_b);
		//status |= clSetKernelArg(clKernel, 4, sizeof(cl_mem), (void *)&d_y);
		//status |= clSetKernelArg(clKernel, 5, sizeof(cl_mem), (void *)&d_Acurr);
		//status |= clSetKernelArg(clKernel, 6, sizeof(int), (void *)&N);
		//status |= clSetKernelArg(clKernel, 6, sizeof(int), (void *)&height_vector);
	
		if (status != CL_SUCCESS)
			printf("clSetKernelArg error(%d)\n", status);
		
	
		//localWorkSize[0] = BLOCK_SIZE;
		//localWorkSize[1] = BLOCK_SIZE;
		//globalWorkSize[0] = width_A;
		//globalWorkSize[1] = height_A;
		localWorkSize[0] = N/BLOCK_SIZE; //1;
		globalWorkSize[0] = (N-n-1)*N/BLOCK_SIZE; //N-n-1;

		status = clEnqueueWriteBuffer(clCommandQue, d_denom, CL_FALSE, 0, mem_size_denom, h_denom, 0, NULL, NULL);
		//status = clEnqueueWriteBuffer(clCommandQue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0, NULL, NULL);
		//status = clEnqueueWriteBuffer(clCommandQue, d_L, CL_FALSE, 0, mem_size_L, h_L, 0, NULL, NULL);
		//status = clEnqueueWriteBuffer(clCommandQue, d_b, CL_FALSE, 0, mem_size_b, h_b, 0, NULL, NULL);
		//status = clEnqueueWriteBuffer(clCommandQue, d_y, CL_FALSE, 0, mem_size_y, h_y, 0, NULL, NULL);
		//status = clEnqueueWriteBuffer(clCommandQue, d_Acurr, CL_FALSE, 0, mem_size_Acurr, h_Acurr, 0, NULL, NULL);
		//printf("Enter the dragon\n");
		status = clEnqueueNDRangeKernel(clCommandQue, 
				clKernel, 1, NULL, globalWorkSize, 
				localWorkSize, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			printf("clEnqueueNDRangeKernel error(%d)\n", status);
		//printf("Exit the dragon\n");
		// 8. Retrieve result from device
		status = clEnqueueReadBuffer(clCommandQue, d_nextDenom, CL_TRUE, 0, mem_size_nextDenom, h_nextDenom, 0, NULL, NULL);
		//status = clEnqueueReadBuffer(clCommandQue, d_x, CL_TRUE, 0, mem_size_x, h_x, 0, NULL, NULL);
		//status = clEnqueueReadBuffer(clCommandQue, d_A, CL_TRUE, 0, mem_size_A, h_A, 0, NULL, NULL);
		//status = clEnqueueReadBuffer(clCommandQue, d_L, CL_TRUE, 0, mem_size_L, h_L, 0, NULL, NULL);
		//printf("HERE2\n");
		if (status != CL_SUCCESS)
			printf("clEnqueueReadBuffer error(%d)\n", status);
		//printf("HERE1\n");
		//printf("nextDenom: %f\n", *nextDenom);
		*denom = *nextDenom;
		//printf("ITERATION %d COMPLETE\n", n);
		
	}
	status = clEnqueueReadBuffer(clCommandQue, d_A, CL_TRUE, 0, mem_size_A, h_A, 0, NULL, NULL);
	
	//printf("HERE2\n");
	if (status != CL_SUCCESS)
		printf("clEnqueueReadBuffer error(%d)\n", status);
	//show_matrix(A,0,N);
	
	//printf("HERE4\n");
	
	// FORWARD SUBSTITUTION
	status = clEnqueueWriteBuffer(clCommandQue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0, NULL, NULL);
	for (n = 0; n < N; n++)
	{
		localWorkSize[0] = N/BLOCK_SIZE_SUB;
		globalWorkSize[0] = N/BLOCK_SIZE_SUB;
		
		status |= clSetKernelArg(clKernelFSub, 3, sizeof(int), (void *)&n);
		if (status != CL_SUCCESS)
			printf("clSetKernelArg error(%d)\n", status);
			
		status = clEnqueueWriteBuffer(clCommandQue, d_y, CL_FALSE, 0, mem_size_y, h_y, 0, NULL, NULL);

		status = clEnqueueNDRangeKernel(clCommandQue, 
				clKernelFSub, 1, NULL, globalWorkSize, 
				localWorkSize, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			printf("clEnqueueNDRangeKernel error(%d)\n", status);
			

		status = clEnqueueReadBuffer(clCommandQue, d_yPart, CL_TRUE, 0, mem_size_yPart, h_yPart, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			printf("clEnqueueReadBuffer error(%d)\n", status);
	
		float sum = 0;
		for (i = 0; i < N/BLOCK_SIZE_SUB; i++)
		{
			sum += yPart[i];
		}
		y[n] = b[n] - sum;
	}
	
	// BACKWARD SUBSTITUTION
	status = clEnqueueWriteBuffer(clCommandQue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0, NULL, NULL);
	for (n = N-1; n >= 0; n--)
	{
		localWorkSize[0] = N/BLOCK_SIZE_SUB;
		globalWorkSize[0] = N/BLOCK_SIZE_SUB;
		
		status |= clSetKernelArg(clKernelBSub, 3, sizeof(int), (void *)&n);
		if (status != CL_SUCCESS)
			printf("clSetKernelArg error(%d)\n", status);
			
		status = clEnqueueWriteBuffer(clCommandQue, d_x, CL_FALSE, 0, mem_size_x, h_x, 0, NULL, NULL);

		status = clEnqueueNDRangeKernel(clCommandQue, 
				clKernelBSub, 1, NULL, globalWorkSize, 
				localWorkSize, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			printf("clEnqueueNDRangeKernel error(%d)\n", status);
			

		status = clEnqueueReadBuffer(clCommandQue, d_xPart, CL_TRUE, 0, mem_size_xPart, h_xPart, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			printf("clEnqueueReadBuffer error(%d)\n", status);
			
		float sum = 0;
		for (i = 0; i < N/BLOCK_SIZE_SUB; i++)
		{
			sum += xPart[i];
		}
		x[n] = (y[n] - sum)/A[n*N+n];
	}
	
	// TEMPORARILY ADDED IN FOR DEBUGGING PURPOSES
	/*for(i = 0; i < N; i++)
	{
		float yi = b[i];
		for(j = 0; j < i; j++)
		{
			yi -= A[i*N+j]*y[j];
		}	
		y[i] = yi;
		
	}
		
	// Use back substitution to solve Ux = y
	for(i = N-1; i >= 0; i--)
	{
		float xi = y[i];
		for(j = i+1; j < N; j++)
			xi -= A[i*N+j]*x[j];
		x[i] = xi/A[i*N+i];
	}
	*/
	// END TEMPORARILY ADDED IN
	
	//printf("HERE5\n");
	//show_matrix(b,0,N);
	//show_matrix(b0,0,N);
	//show_matrix(x,0,N);
	// stop timer
	clock_t end = clock();
	gettimeofday(&t2, NULL);
    timersub(&t1, &t2, &tr);
	
	elapsed += ((double)(end-start)) / CLOCKS_PER_SEC;
	
	// Check result
	double error = 0;
	for(i = 0; i < N; i++)
	{
		double b_res = 0;
		for(j = 0; j < N; j++)
			b_res += A0[i*N+j] * x[j];
		// correctness is perfect for doubles, but the lack of precision in floats will cause values in the calculated vector
		// to not be exactly the same as the original b vector
		if ( ((b_res < b0[i]) && ((b_res + (N*0.5)) >= b0[i])) || ((b_res > b0[i]) && ((b_res - (N*0.5)) <= b0[i])) )
			b_res = b0[i];
		error += b_res > b0[i] ? b_res-b0[i] : b0[i]-b_res;
		//printf("b_res is: %f\n", b_res);
	}
		
	double epsilonPerRow = 0.01;
	if(error < N*epsilonPerRow)
		passed++;
		
	printf("Error: %f\n", error);
	printf("%d of %d tests passed\n", passed, ITER);
	printf("Average time (like in mmul): %.2f seconds\n", (fabs(tr.tv_sec+(double)tr.tv_usec/1000000.0))/ITER);
	printf("Average time: %.2f seconds\n", elapsed/ITER);

	// 10. clean up memory
	free(A0);
	free(b0);
	
	free(h_A);
	free(h_L);
	free(h_b);
	free(h_x);
	free(h_y);
	free(h_Acurr);
	free(h_xPart);
	free(h_yPart);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_L);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_y);
	clReleaseMemObject(d_Acurr);
	clReleaseMemObject(d_xPart);
	clReleaseMemObject(d_yPart);

	free(devices);
	clReleaseContext(context);
	clReleaseKernel(clKernel);
	clReleaseKernel(clKernelFSub);
	clReleaseKernel(clKernelBSub);
	clReleaseProgram(program);
	clReleaseCommandQueue(clCommandQue);
}
	
	
	
	
