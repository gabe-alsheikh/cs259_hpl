#define N 4

__kernel void
LUFact(
	__global float* x,
	__global float* A,
	__global float* L,
	__global float* b,
	__global float* y)
{
	// Thread/work item index within group (represents rows)
	int tIndex = get_local_id(0);
	
	// Local copies of the rows of A and L
	//float At[N];
	//float Lt[N];
	//for(int i = 0; i < N; i++)
    //At[i] = A[tIndex*N+i];
	
	local float denom;
	__local float Acurr[N];
	if (tIndex == 0) {
		for(int i = 0; i < N; i++)
			Acurr[i] = A[tIndex*N+i];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // GLOBAL? (if change structure)

	L[tIndex*N+tIndex] = 1;
	for(int n = 0; n < N; n++) {
		denom = Acurr[n];
		if(tIndex > n) {
			float lval = A[tIndex*N+n]/denom;
			L[tIndex*N+n] = lval;
			lval = -lval;
			for(int i = n+1; i < N; i++) {
				A[tIndex*N+i] += lval*Acurr[i];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); // may be necessary
		if(tIndex == n+1) {
			for(int i = tIndex; i < N; i++)
				Acurr[i] = A[tIndex*N+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE); // should be necessary	
	}

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	float yi = b[tIndex];
	for(int i = 0; i < N; i++) {
		if(i == tIndex) {
			for(int j = 0; j < tIndex; j++) {
				yi -= L[tIndex*N+j]*y[j];
			}	
			y[tIndex] = yi;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Use back substitution to solve Ux = y
	float xi = y[tIndex];
	for(int i = N-1; i >= 0; i--) {
		if(i == tIndex) {
			for(int j = tIndex+1; j < N; j++)
				xi -= A[tIndex*N+j]*x[j];
			x[tIndex] = xi/A[tIndex*N+tIndex];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
