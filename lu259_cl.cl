__kernel void
LUFact(
	__global double* x,
	__global double* A, 
	__global double* L, 
	__global double* b,
	__global double* y,
	int width_matrix, int height_vector)
{
	// Thread/work item index within group (represents rows)
	int tIndex = get_local_id(0);
	

	__local double denom;
	if (tIndex == 0)
		denom = A[tIndex*width_matrix+tIndex];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	L[tIndex*width_matrix+tIndex] = 1;
	for (int i = tIndex+1; i < width_matrix; i++)
	{
		double lval = A[i*width_matrix+tIndex]/denom;
		L[i*width_matrix+tIndex] = lval;
		lval = -lval;
		for(int k = tIndex+1; k < width_matrix; k++)
		{
			A[i*width_matrix+k] += lval*A[tIndex*width_matrix+k];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	double yi = b[tIndex];
	for(int j = 0; j < tIndex; j++)
	{
		yi -= L[tIndex*width_matrix+j]*y[j];
	}	
	y[tIndex] = yi;
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Use back substitution to solve Ux = y
	double xi = y[tIndex];
	for(int j = tIndex+1; j < width_matrix; j++)
		xi -= A[tIndex*width_matrix+j]*x[j];
	x[tIndex] = xi/A[tIndex*width_matrix+tIndex];
	
	barrier(CLK_LOCAL_MEM_FENCE);
}
