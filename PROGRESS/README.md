This folder will detail the primarily line of progression of the project, and will include associated files in this process.

First change: Moved writeBuffer and readBuffer out of the iterative loop for 
the LU decomposition kernel call.

lu259_cl_ITERKERNEL_v1.X
This iteration involved a kernel with N-n-1 work groups and an iterative scheme to
perform the LU decomposition. Each work group is a single item. Substitution is
done in the host file.
SUCCESSFULLY TESTED ON FPGA USING FLOATS

lu259_cl_ITERKERNEL_v2.X
Converted the previous kernel into a blocked scheme, with each work group having
N/BLOCK_SIZE items (global size is (N-n-1)*N/BLOCK_SIZE)
No speedup over v1, but it could potentially provide it with HLS optimizations.
