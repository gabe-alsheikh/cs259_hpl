__kernel void
LUFact(
	__global double* A,
	int n,
	int N)
{
	// N-n-1 work-groups, 1 work-item each
	int row = get_group_id(0) + n+1;
	// load rows n, row?
	double denom = A[n*N+n];
	double lval = A[row*N+n]/denom;
	A[row*N+n] = lval;
	lval = -lval;
	for(int i = n+1; i < N; i++)
		A[row*N+i] += lval*A[n*N+i]; 
		// overlap comm and comp?
	// store local row?
}
