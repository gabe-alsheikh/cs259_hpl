#define N 512

__kernel void
LUFact(
	__global double* x,
	__global double* A,
	__global double* L,
	__global double* b,
	__global double* y)
{
	// Thread/work item index within group (represents rows)
	int tIndex = get_local_id(0);
	
	// Local copies of the rows of A and L
	double At[N];
	double Lt[N];
	for(int i = 0; i < N; i++)
		At[i] = A[tIndex*N+i];

	double denom;
	__local double Acurr[N];
	if (tIndex == 0) {
		for(int i = 0; i < N; i++)
			Acurr[i] = At[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // GLOBAL? (if change structure)

	Lt[tIndex] = 1;
	for(int n = 0; n < N; n++) {
		denom = Acurr[n];
		if(tIndex > n) {
			double lval = At[n]/denom;
			Lt[n] = lval;
			lval = -lval;
			for(int i = n+1; i < N; i++) {
				At[i] += lval*Acurr[i];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); // may be necessary
		if(tIndex == n+1) {
			for(int i = tIndex; i < N; i++)
				Acurr[i] = At[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE); // should be necessary	
	}

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	double yi = b[tIndex];
	for(int i = 0; i < N; i++) {
		if(i == tIndex) {
			for(int j = 0; j < tIndex; j++) {
				yi -= Lt[j]*y[j];
			}	
			y[tIndex] = yi;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Use back substitution to solve Ux = y
	double xi = y[tIndex];
	for(int i = N-1; i >= 0; i--) {
		if(i == tIndex) {
			for(int j = tIndex+1; j < N; j++)
				xi -= At[j]*x[j];
			x[tIndex] = xi/At[tIndex];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
