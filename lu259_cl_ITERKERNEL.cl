#define BLOCK_SIZE 128
#define BLOCK_SIZE_SUB 128

__kernel void
LUFact(
	__global float* A,
	__global float* denom,
	__global float* nextDenom,
	int n,
	int N)
{
	// N-n-1 work-groups, N/BLOCK_SIZE work-items each
	int row = get_group_id(0) + n+1;
	int item = get_local_id(0);
	int cStart = (n/BLOCK_SIZE + item) * BLOCK_SIZE;
	int cEnd = cStart + BLOCK_SIZE;
	int rStart = row*N;
	int nStart = n*N;
	// load rows n, row?
	float lval = A[rStart+n]/(*denom);
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(item == 0)
		A[rStart+n] = lval;
	lval = -lval;
	int i = (n+1) < cStart ? cStart : (n+1);
	if(i > cStart && i < cEnd)
	{
		for(; i%4 != 0; i++)
			A[rStart+i] += lval*A[nStart+i];
	}
	__attribute__((xcl_pipeline_loop))
	for(; i < cEnd; i+=4)
	{
		A[rStart+i] += lval*A[nStart+i];
		A[rStart+i+1] += lval*A[nStart+i+1];
		A[rStart+i+2] += lval*A[nStart+i+2];
		A[rStart+i+3] += lval*A[nStart+i+3];
	}
	if(row == n+1 && cStart <= n+1 && cEnd > n+1)
		*nextDenom = A[rStart+n+1];
}
