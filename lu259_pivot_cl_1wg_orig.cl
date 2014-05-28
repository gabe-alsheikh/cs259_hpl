__kernel void
LUFact(
	__global double* x,
	__global double* A, 
	__global double* L, 
	__global double* b,
	__global double* y,
	int width_matrix, height_vector)
{
	// Thread/work item index within group
	int tIndex = get_local_id(0);
	
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// !!!!!IMPORTANT NOTE: WE ONLY HAVE ONE WORK GROUP!!!!!
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	
	// Each work group should contain one block
	// eg. if block size is 16, then each group has 16 work items
	// Actual global start position of the group in matrix

	__local double denom;
	if (tIndex == 0)
		denom = A[tIndex*width_matrix+tIndex];
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	L[tIndex*width_matrix+tIndex] = 1;
	for (int i = tIndex+1; i < matrix_width; i++)
	{
		double lval = A[i*matrix_width+tIndex]/denom;
		L[i*matrix_width+tIndex] = lval;
		lval = -lval;
		for(int k = tIndex+1; k < matrix_width; k++)
		{
			A[i*matrix_width+k] += lval*A[tIndex*matrix_width+k];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	double yi = b[tIndex];
	for(int j = 0; j < tIndex; j++)
	{
		yi -= L[tIndex*matrix_width+j]*y[j];
	}	
	y[tIndex] = yi;
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Use back substitution to solve Ux = y
	double xi = y[tIndex];
	for(int j = tIndex+1; j < matrix_width; j++)
		xi -= A[tIndex*matrix_width+j]*x[j];
	x[tIndex] = xi/A[tIndex*matrix_width+tIndex];
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	
}
