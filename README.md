# StencilBenchmarks
  
  Star stencil CUDA kernels.
  
  NVIDIA implementation from  https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf

  + `kernel_nvidia.cu` Block 16x16, Order 8, radius 4
  + `kernel_nvidia_4.cu` Block 32x32, Order 8, radius 4
  + `kernel_nvidia_32.cu` Block 32x32, Order 16, radius 8
  
  
  
  Improved 
  
  + `my_kernel_regs.cu` Block 16x16, Order 8, radius 4
  + `my_kernel_regs_4.cu` Block 32x32, Order 8, radius 4
  + `my_kernel_regs_32.cu` Block 32x32, Order 16, radius 8
  
  
  ## Run
  
  `./mainProgram x y z`
  
  x y z are the length for each and is raised to the power of 2
