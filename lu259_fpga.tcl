# Create Project 
create_project -name lu259_fpga -platform zc706-linux-uart -force

# Add host code
add_host_src -filename "lu259_cl.c" -type source

# Create a kernel for the LU factorization
create_kernel -id LUFact -type clc
add_kernel_src -id LUFact -filename "lu259_cl.cl"
create_kernel -id fSub -type clc
add_kernel_src -id fSub -filename "lu259_cl.cl"
create_kernel -id bSub -type clc
add_kernel_src -id bSub -filename "lu259_cl.cl"

# Create a Xilinx OpenCL Kernel Binary Container
create_xclbin lu259_fpga

# Select the execution target of the kernel
# EDIT: -N 8 parameter added in 
map_add_kernel_instance -xclbin lu259_fpga -id LUFact -target fpga0:OCL_REGION_0
map_add_kernel_instance -xclbin lu259_fpga -id fSub -target fpga0:OCL_REGION_0
map_add_kernel_instance -xclbin lu259_fpga -id bSub -target fpga0:OCL_REGION_0


# Compile the host code
compile_host -arch arm -cflags "-O3 -Wall -D FPGA_DEVICE -D TEST"

# Build the system
#build_estimate
build_system

# Package SD Card Image
#build_sdimage

exit
