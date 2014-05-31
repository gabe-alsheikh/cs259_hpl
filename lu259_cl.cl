#define BLOCK_SIZE 64 // to be decreased later

__kernel void
LUFact(
	__global double* x,
	__global double* A,
	__global double* L,
	__global double* b,
	__global double* y,
	__global double* Acurr,
	int N)
{
	// One work group per row (N total)
	// N/BLOCK_SIZE work items per group
	int row = get_group_id(0);
	int tIndex = get_local_id(0);
	
	int cStart = BLOCK_SIZE*tIndex;
	int cEnd = cStart+BLOCK_SIZE;

	// Local copies of the rows of A and L
	double At[BLOCK_SIZE];
	double Lt[BLOCK_SIZE];

	for(int i = cStart; i < cEnd; i++)
		At[i] = A[tIndex*N+i];

	double denom;
	__local double lval;
	//__global double Acurr[N];
	for(int n = 0; n < N; n++) {
		if(row == n && cEnd > n) {
			int c = n < cStart ? cStart : n;
			for(; c < cEnd; c++)
				Acurr[c] = At[c-cStart];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);

		denom = Acurr[n];
		if(row > n) {
			if(n >= cStart && n < cEnd) {
				lval = Acurr[n]/denom;
				Lt[n-cStart] = lval;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			lval = -lval;
			int i = n+1 < cStart ? cStart : n+1;
			for(; i < cEnd; i++) {
				At[i-cStart] += lval*Acurr[i];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);	
		//if (row == 0 && tIndex == 3)
		//printf(\"HERE2\");
	}

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	__local double yi;
    	yi = b[row];
	double sum = 0;
	for(int i = 0; i < N; i++) {
		if(i == row) {
			int e = row < cEnd ? row : cEnd;
			for(int j = cStart; j < row && j < cEnd; j++)
				sum += Lt[j-cStart]*y[j];
			yi -= sum;
			barrier(CLK_LOCAL_MEM_FENCE);
			async_work_group_copy(y+row, &yi, 1, 0);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Use back substitution to solve Ux = y
	__local double xi;
    	xi = y[tIndex];
	sum = 0;
	for(int i = N-1; i >= 0; i--) {
		if(i == row) {
			int j = row+1 < cStart ? cStart : row+1;
			for(; j < cEnd; j++)
				sum += At[j-cStart]*x[j];
			xi -= sum;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(cStart <= row && row < cEnd)
				x[row] = xi/At[row-cStart];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
