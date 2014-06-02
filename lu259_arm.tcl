# Create Project 
create_project -name lu259_arm -platform zc706-linux-uart -force

# Add host code 
add_host_src -filename "lu259_cl.c" -type source

# Compile the host code
compile_host -arch arm -cflags "-O3 -Wall -D FPGA_DEVICE"

# Build xclbin
build_system

exit
