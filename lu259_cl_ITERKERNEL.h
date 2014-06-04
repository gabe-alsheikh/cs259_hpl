const char* lu259_cl =
"#define BLOCK_SIZE 64\n"
"__kernel void\n"
"LUFact(\n"
"__global double* A,\n"
"__global double* denom,\n"
"__global double* nextDenom,\n"
"int n,\n"
"int N)\n"
"{\n"
"int row = get_group_id(0) + n+1;\n"
"int item = get_local_id(0);\n"
"int cStart = item * BLOCK_SIZE;\n"
"int cEnd = cStart + BLOCK_SIZE;\n"
"//double denom = A[n*N+n];\n"
"double lval = A[row*N+n]/(*denom);\n"
"barrier(CLK_GLOBAL_MEM_FENCE);\n"
"if (item == 0)\n"
"A[row*N+n] = lval;\n"
"lval = -lval;\n"
"int i = (n+1) < cStart ? cStart : (n+1);\n"
"for(; i < cEnd; i++)\n"
"A[row*N+i] += lval*A[n*N+i]; \n"
"if(row == (n+1) && item == 0)\n"
"*nextDenom = A[(n+1)*N+n+1];\n"
"}\n"
;
