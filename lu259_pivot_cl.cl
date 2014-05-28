// Thread block size
#define BLOCK_SIZE 16
  
__kernel void
LUFact(
	__global double* x,
	__global double* A, 
	__global double* L, 
	__global double* b,
	__global double* y,
	int width_matrix, height_vector)
{

	// Group index within global
	int gIndex = get_group_id(0);

	// Thread/work item index within group
	int tIndex = get_local_id(0);
	
	
	// Each work group should contain one block
	// eg. if block size is 16, then each group has 16 work items
	// Actual global start position of the group in matrix
	int groupStart = gIndex*BlOCK_SIZE*BLOCK_SIZE;
	
	// Actual global position of the thread in matrix
	int globalIndex = groupStart + tIndex;

	double denom = A[globalIndex*width_matrix+globalIndex];
	
	L[globalIndex*width_matrix+globalIndex] = 1;
	for (int i = globalIndex+1; i < matrix_width; i++)
	{
		double lval = A[i*matrix_width+globalIndex]/denom;
		L[i*matrix_width+globalIndex] = lval;
		lval = -lval;
		for(int k = globalIndex+1; k < matrix_width; k++)
		{
			A[i*matrix_width+k] += lval*A[globalIndex*matrix_width+k];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	double yi = b[globalIndex];
	for(int j = 0; j < globalIndex; j++)
	{
		yi -= L[globalIndex*matrix_width+j]*y[j];
	}	
	y[globalIndex] = yi;
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Use back substitution to solve Ux = y
	double xi = y[globalIndex];
	for(int j = globalIndex+1; j < matrix_width; j++)
		xi -= A[globalIndex*matrix_width+j]*x[j];
	x[globalIndex] = xi/A[globalIndex*matrix_width+globalIndex];
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	
}
