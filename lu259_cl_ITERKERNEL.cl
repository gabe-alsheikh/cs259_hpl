#define BLOCK_SIZE 64

__kernel void
LUFact(
	__global double* A,
	int n,
	int N)
{
	// N-n-1 work-groups, 1 work-item each
	int row = get_group_id(0) + n+1;
	int item = get_local_id(0);
	int cStart = item * BLOCK_SIZE;
	int cEnd = cStart + BLOCK_SIZE;
	// load rows n, row?
	double denom = A[n*N+n];
	double lval = A[row*N+n]/denom;
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (item == 0)
		A[row*N+n] = lval;
	lval = -lval;
	int i = (n+1) < cStart ? cStart : (n+1);
	for(; i < cEnd; i++)
		A[row*N+i] += lval*A[n*N+i]; 
		// overlap comm and comp?
	// store local row?
}
