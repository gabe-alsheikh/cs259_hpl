#define BLOCK_SIZE 16
#define BLOCK_SIZE_SUB 16
__kernel void
LUFact(
	__global float* A,
	__global float* lval, // an array of size N
	__global float* nextLval,
	__global float* nextDenom,
	int n,
	int N)
{
	// ceil((N-n-1)/BLOCK_SIZE)-by-ceil((N-n-1)/BLOCK_SIZE) work-groups
	// BLOCK_SIZE work-items per group
	int groupsPerRow = (N-n-2)/BLOCK_SIZE + 1;
	int g = get_group_id(0);
	int brStart = BLOCK_SIZE * ((n+1)/BLOCK_SIZE + (g/groupsPerRow));
	int t = get_local_id(0);
	int row = brStart + t;
	int cStart = BLOCK_SIZE * ((n+1)/BLOCK_SIZE + (g % groupsPerRow));
	int cEnd = cStart + BLOCK_SIZE;
	int rStart = row*N;
	int nStart = n*N;
	/*__local float Lcurr[BLOCK_SIZE];
	__local float Acurr[BLOCK_SIZE];
	async_work_group_copy(Lcurr, lval+brStart, BLOCK_SIZE, 0);
	async_work_group_copy(Acurr, A+nStart+cStart, BLOCK_SIZE, 0);*/
	float l = lval[row];//Lcurr[t];
	barrier(CLK_GLOBAL_MEM_FENCE); // may not be needed
	if(row >= n+1)
	{
		float lneg = -l;
		int i = (n+1) < cStart ? cStart : (n+1);
		if(i > cStart && i < cEnd)
			for(; i%4 != 0; i++)
				A[rStart+i] += lneg*A[nStart+i];
		/*int i = (n+1) < cStart ? 0 : (n+1-cStart);
		if(i > 0)
			for(; i%4 != 0; i++)
				A[rStart+cStart+i] += lneg*Acurr[i];*/
		// pipeline here
		__attribute__((xcl_pipeline_loop))
		for(; i < cEnd; i+=4)
		{
			A[rStart+i] += lneg*A[nStart+i];
			A[rStart+i+1] += lneg*A[nStart+i+1];
			A[rStart+i+2] += lneg*A[nStart+i+2];
			A[rStart+i+3] += lneg*A[nStart+i+3];
		}
		/*for(; i < BLOCK_SIZE; i+=4)
		{
			A[rStart+cStart+i] += lneg*Acurr[i];
			A[rStart+cStart+i+1] += lneg*Acurr[i+1];
			A[rStart+cStart+i+2] += lneg*Acurr[i+2];
			A[rStart+cStart+i+3] += lneg*Acurr[i+3];
		}*/
		if(cStart <= n+1 && cEnd > n+1)
		{
			A[rStart+n] = l;
			nextLval[row] = A[rStart+n+1]; // use this and nextDenom to generate next lvals
			if(row == n+1)
				*nextDenom = A[rStart+n+1];
		}
	}
}
