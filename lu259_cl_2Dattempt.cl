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
	// Thread/work item index within group (represents rows)
	int tX = get_local_id(0); // across columns
	int tY = get_local_id(1); // rows
	// tIndex is now tY
	
	// Local copies of the rows of A and L
	//double At[N];
	//double Lt[N];
	//for(int i = 0; i < N; i++)
    //At[i] = A[tIndex*N+i];
	
	double denom;
	//__local double Acurr[N];
	if (tY == 0) {
			Acurr[tx] = A[tY*N+tX];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // GLOBAL? (if change structure)

	if (tX == tY)
		L[tY*N+tY] = 1;
	barrier(CLK_LOCAL_MEM_FENCE);
		denom = Acurr[tX];
		if(tY > tX) {
			double lval = A[tY*N+tX]/denom;
			L[tY*N+tX] = lval;
			lval = -lval;
			for(int i = tX+1; i < N; i++) {
				A[tY*N+i] += lval*Acurr[i];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); // may be necessary
		if(tY == tX+1) {
			for(int i = tY; i < N; i++)
				Acurr[i] = A[tY*N+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE); // should be necessary	
	

	// A(N-1) becomes U
	// Use forward substitution to solve Ly = Pb
	double yi = b[tY];
	for(int i = 0; i < N; i++) {
		if(i == tY) {
			for(int j = 0; j < tY; j++) {
				yi -= L[tY*N+j]*y[j];
			}	
			y[tY] = yi;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Use back substitution to solve Ux = y
	double xi = y[tY];
	for(int i = N-1; i >= 0; i--) {
		if(i == tY) {
			for(int j = tY+1; j < N; j++)
				xi -= A[tY*N+j]*x[j];
			x[tY] = xi/A[tY*N+tY];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
