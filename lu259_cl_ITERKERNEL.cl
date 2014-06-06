#define BLOCK_SIZE 32
#define BLOCK_SIZE_SUB 32

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
	int cStart = item * BLOCK_SIZE;
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
		for(; i%4 != 0; i++)
			A[rStart+i] += lval*A[nStart+i];
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

__kernel void
fSub(
	__global float* A,
	__global float* y,
	__global float* yPart,
	int n,
	int N)
{
	// N/BLOCK_SIZE w-g's or w-i's
	int t = get_global_id(0);
	int cStart = t*BLOCK_SIZE_SUB;
	int cEnd = cStart+BLOCK_SIZE_SUB;
	float pSum = 0;
	int e = cEnd < n ? cEnd : n;
	for(int i = cStart; i < e; i++)
		pSum += A[n*N+i]*y[i];
	yPart[t] = pSum;
}


__kernel void
bSub(
	__global float* A,
	__global float* x,
	__global float* xPart,
	int n,
	int N)
{
	int t = get_global_id(0);
	int cStart = t*BLOCK_SIZE_SUB;
	int cEnd = cStart+BLOCK_SIZE_SUB;
	float pSum = 0;
	int i = (n+1) < cStart ? cStart : (n+1);
	for(; i < cEnd; i++)
		pSum += A[n*N+i]*x[i];
	xPart[t] = pSum;
}
