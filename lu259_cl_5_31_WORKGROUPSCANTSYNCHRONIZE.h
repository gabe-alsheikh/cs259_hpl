const char* lu259_cl =
"#define BLOCK_SIZE 64 // to be decreased later\n"
"__kernel void\n"
"LUFact(\n"
"__global double* x,\n"
"__global double* A,\n"
"__global double* L,\n"
"__global double* b,\n"
"__global double* y,\n"
"__global double* Acurr,\n"
"int N)\n"
"{\n"
"int row = get_group_id(0);\n"
"int tIndex = get_local_id(0);\n"
"int cStart = BLOCK_SIZE*tIndex;\n"
"int cEnd = cStart+BLOCK_SIZE;\n"
"double At[BLOCK_SIZE];\n"
"double Lt[BLOCK_SIZE];\n"
"for(int i = cStart; i < cEnd; i++)\n"
"At[i-cStart] = A[row*N+i];\n"
"double denom;\n"
"__local double lval;\n"
"//__global double Acurr[N];\n"
"for(int n = 0; n < N; n++) {\n"
"if(row == n) {\n"
"//int c = n < cStart ? cStart : n;\n"
"int c = cStart;\n"
"for(; c < cEnd; c++)\n"
"Acurr[c] = At[c-cStart];\n"
"}\n"
"barrier(CLK_GLOBAL_MEM_FENCE);\n"
"//denom = Acurr[n];\n"
"if(row > n) {\n"
"if(n >= cStart && n < cEnd) {\n"
"denom = Acurr[n];\n"
"lval = At[n-cStart]/denom;\n"
"Lt[n-cStart] = lval;\n"
"}\n"
"else if(row == n && n >= cStart && n < cEnd)\n"
"Lt[n-cStart] = 1;\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"//lval = -lval;\n"
"int i = (n+1) < cStart ? cStart : (n+1);\n"
"for(; i < cEnd; i++) {\n"
"At[i-cStart] -= lval*Acurr[i];\n"
"}\n"
"}\n"
"barrier(CLK_GLOBAL_MEM_FENCE);\n"
"//if (row == 0 && tIndex == 3)\n"
"//printf(\"HERE2\");\n"
"}\n"
"for(int i = cStart; i < cEnd; i++)\n"
"{\n"
"A[row*N+i] = At[i];\n"
"L[row*N+i] = Lt[i];\n"
"}\n"
"barrier(CLK_GLOBAL_MEM_FENCE);\n"
"__local double yi;\n" 
"yi = b[row];\n"
"double sum = 0;\n"
"for(int i = 0; i < N; i++) {\n"
"if(i == row) {\n"
"int e = row < cEnd ? row : cEnd;\n"
"for(int j = cStart; j < e; j++)\n"
"sum += Lt[j-cStart]*y[j];\n"
"yi -= sum;\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"async_work_group_copy(y+row, &yi, 1, 0);\n"
"}\n"
"barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
"__local double xi;\n" 
"xi = y[tIndex];\n"
"sum = 0;\n"
"for(int i = N-1; i >= 0; i--) {\n"
"if(i == row) {\n"
"int j = (row+1) < cStart ? cStart : (row+1);\n"
"for(; j < cEnd; j++)\n"
"sum += At[j-cStart]*x[j];\n"
"xi -= sum;\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if(cStart <= row && row < cEnd)\n"
"x[row] = xi/At[row-cStart];\n"
"}\n"
"barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
"}\n"
;
