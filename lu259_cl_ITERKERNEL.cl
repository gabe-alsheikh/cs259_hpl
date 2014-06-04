#define BLOCK_SIZE 64
#define BLOCK_SIZE_SUB 64

__kernel void
LUFact(
	__global double* A,
	__global double* denom,
	__global double* nextDenom,
	int n,
	int N)
{
	// N-n-1 work-groups, 1 work-item each
	int row = get_group_id(0) + n+1;
	int item = get_local_id(0);
	int cStart = item * BLOCK_SIZE;
	int cEnd = cStart + BLOCK_SIZE;
	// load rows n, row?
	double lval = A[row*N+n]/(*denom);
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (item == 0)
		A[row*N+n] = lval;
	lval = -lval;
	int i = (n+1) < cStart ? cStart : (n+1);
	for(; i < cEnd; i++)
		A[row*N+i] += lval*A[n*N+i]; 
	if(row == (n+1) && cStart <= n+1 && cEnd > n+1)
		*nextDenom = A[(n+1)*N+n+1];
		// overlap comm and comp?
	// store local row?
}

__kernel void
fSub(
	__global double* A,
	__global double* y,
	__global double* yPart,
	int n,
	int N)
{
	// N/BLOCK_SIZE w-g's or w-i's
	int t = get_global_id(0);
	int cStart = t*BLOCK_SIZE_SUB;
	int cEnd = cStart+BLOCK_SIZE_SUB;
	double pSum = 0;
	int e = cEnd < n ? cEnd : n;
	for(int i = cStart; i < e; i++)
		pSum += A[n*N+i]*y[i];
	yPart[t] = pSum;
}


__kernel void
bSub(
	__global double* A,
	__global double* x,
	__global double* xPart,
	int n,
	int N)
{
	int t = get_global_id(0);
	int cStart = t*BLOCK_SIZE_SUB;
	int cEnd = cStart+BLOCK_SIZE_SUB;
	double pSum = 0;
	int i = (n+1) < cStart ? cStart : (n+1);
	for(; i < cEnd; i++)
		pSum += A[n*N+i]*x[i];
	xPart[t] = pSum;
}
