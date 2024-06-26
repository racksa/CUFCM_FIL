// fcm_mobility_solver.cu

#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "fcm_mobility_solver.hpp"
#include "../cuda_functions.hpp"

fcm_mobility_solver::~fcm_mobility_solver(){}

fcm_mobility_solver::fcm_mobility_solver(){
  Real values[100];
  std::string fcm_folder = "";
  std::vector<std::string> datafile_names{3};
  read_config(values, datafile_names, CUFCM_CONFIG_FILE_NAME.c_str());
  for(int i = 0; i < 3; i++){
    datafile_names[i] = fcm_folder + datafile_names[i];
  }

  pars.N = NSWIM*NFIL*NSEG + NSWIM*NBLOB;
  pars.rh = values[1];
  pars.alpha = values[2];
  pars.beta = values[3];
  pars.eta = values[4];
  pars.nx = values[5];
  pars.ny = values[6];
  pars.nz = values[7];
  pars.repeat = values[8];
  pars.prompt = values[9];
  pars.boxsize = values[13];

  // Overwriting the parameters from globals.ini
  pars.nx = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "nx"));
  pars.ny = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "ny"));
  pars.nz = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "nz"));
  pars.boxsize = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "boxsize"));

  cufcm_solver = new FCM_solver(pars);
  cufcm_solver->init_aux_for_filament();
  cufcm_solver->num_seg = NSWIM*NFIL*NSEG;
  cufcm_solver->num_blob = NSWIM*NBLOB;
}


void fcm_mobility_solver::free_host_memory(){
  
  delete[] num_segs;
  delete[] num_blobs;

  cudaFreeHost(v_segs_host);
  cudaFreeHost(v_blobs_host);
  cudaFreeHost(x_segs_host);
  cudaFreeHost(x_blobs_host);
  cudaFreeHost(f_segs_host);
  cudaFreeHost(f_blobs_host);
  cudaFreeHost(f_blobs_repulsion_host);

}

void fcm_mobility_solver::free_device_memory(){

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaFree(v_segs_device[n]);
    cudaFree(v_blobs_device[n]);
    cudaFree(x_segs_device[n]);
    cudaFree(x_blobs_device[n]);
    cudaFree(f_segs_device[n]);
    cudaFree(f_blobs_device[n]);
    cudaFree(f_blobs_repulsion_device[n]);

  }

  delete[] v_segs_device;
  delete[] v_blobs_device;
  delete[] x_segs_device;
  delete[] x_blobs_device;
  delete[] f_segs_device;
  delete[] f_blobs_device;
  delete[] f_blobs_repulsion_device;

}

void fcm_mobility_solver::allocate_host_memory(){
  
  std::cout << std::endl << std::endl << "Running on all GPUs visible to this shell environment, as defined by the environment variable CUDA_VISIBLE_DEVICES." << std::endl;

  cudaGetDeviceCount(&num_gpus);

  std::cout <<  "Found " << num_gpus << " GPU(s)." << std::endl;

  cudaSetDevice(0);

  // Allocate pinned host memory to allow async copying
  cudaHostAlloc(&v_segs_host, 6*NSWIM*NFIL*NSEG*sizeof(Real), cudaHostAllocPortable);
  cudaHostAlloc(&v_blobs_host, 3*NSWIM*NBLOB*sizeof(Real), cudaHostAllocPortable);
  cudaHostAlloc(&x_segs_host, 3*NSWIM*NFIL*NSEG*sizeof(Real), cudaHostAllocPortable);
  cudaHostAlloc(&x_blobs_host, 3*NSWIM*NBLOB*sizeof(Real), cudaHostAllocPortable);
  cudaHostAlloc(&f_segs_host, 6*NSWIM*NFIL*NSEG*sizeof(Real), cudaHostAllocPortable);
  cudaHostAlloc(&f_blobs_host, 3*NSWIM*NBLOB*sizeof(Real), cudaHostAllocPortable);
  cudaHostAlloc(&f_blobs_repulsion_host, 3*NSWIM*NBLOB*sizeof(Real), cudaHostAllocPortable);

  num_segs = new int[num_gpus];
  num_blobs = new int[num_gpus];

  num_segs[0] = NSWIM*NFIL*NSEG;
  num_blobs[0] = NSWIM*NBLOB;

  for (int n = 1; n < num_gpus; n++){

    num_segs[n] = (NSWIM*NFIL*NSEG)/num_gpus;
    num_segs[0] -= num_segs[n];

    num_blobs[n] = (NSWIM*NBLOB)/num_gpus;
    num_blobs[0] -= num_blobs[n];

  }

}

void fcm_mobility_solver::allocate_device_memory(){
  
  v_segs_device = new Real*[num_gpus];
  v_blobs_device = new Real*[num_gpus];
  x_segs_device = new Real*[num_gpus];
  x_blobs_device = new Real*[num_gpus];
  f_segs_device = new Real*[num_gpus];
  f_blobs_device = new Real*[num_gpus];
  f_blobs_repulsion_device = new Real*[num_gpus];

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);

    cudaMalloc(&v_segs_device[n], 6*num_segs[n]*sizeof(Real));
    cudaMalloc(&v_blobs_device[n], 3*num_blobs[n]*sizeof(Real));

    cudaMalloc(&x_segs_device[n], 3*NSWIM*NFIL*NSEG*sizeof(Real));
    cudaMalloc(&x_blobs_device[n], 3*NSWIM*NBLOB*sizeof(Real));

    cudaMalloc(&f_segs_device[n], 6*NSWIM*NFIL*NSEG*sizeof(Real));
    cudaMalloc(&f_blobs_device[n], 3*NSWIM*NBLOB*sizeof(Real));
    cudaMalloc(&f_blobs_repulsion_device[n], 3*num_blobs[n]*sizeof(Real));

  }

}

void fcm_mobility_solver::copy_segment_positions_to_device(){

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaMemcpyAsync(x_segs_device[n], x_segs_host, 3*NSWIM*NFIL*NSEG*sizeof(Real), cudaMemcpyHostToDevice);

  }

}

void fcm_mobility_solver::copy_segment_forces_to_device(){

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaMemcpyAsync(f_segs_device[n], f_segs_host, 6*NSWIM*NFIL*NSEG*sizeof(Real), cudaMemcpyHostToDevice);

  }

}

void fcm_mobility_solver::copy_blob_positions_to_device(){

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaMemcpyAsync(x_blobs_device[n], x_blobs_host, 3*NSWIM*NBLOB*sizeof(Real), cudaMemcpyHostToDevice);

  }

}

void fcm_mobility_solver::copy_blob_forces_to_device(){

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaMemcpyAsync(f_blobs_device[n], f_blobs_host, 3*NSWIM*NBLOB*sizeof(Real), cudaMemcpyHostToDevice);

  }

}

void fcm_mobility_solver::copy_interparticle_blob_forces_to_host(){

  int start_blob = 0;

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaMemcpyAsync(&f_blobs_repulsion_host[3*start_blob], f_blobs_repulsion_device[n], 3*num_blobs[n]*sizeof(Real), cudaMemcpyDeviceToHost);
    start_blob += num_blobs[n];

  }

}

void fcm_mobility_solver::copy_blob_velocities_to_host(){

  int start_blob = 0;

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);
    cudaMemcpyAsync(&v_blobs_host[3*start_blob], v_blobs_device[n], 3*num_blobs[n]*sizeof(Real), cudaMemcpyDeviceToHost);
    start_blob += num_blobs[n];

  }

}

void fcm_mobility_solver::copy_segment_velocities_to_host(){

  int start_seg = 0;

  for (int n = 0; n < num_gpus; n++){

    cudaSetDevice(n);

    cudaMemcpyAsync(&v_segs_host[6*start_seg], v_segs_device[n], 6*num_segs[n]*sizeof(Real), cudaMemcpyDeviceToHost);

    start_seg += num_segs[n];

  }

}

void fcm_mobility_solver::apply_interparticle_forces(){

  #if !PRESCRIBED_CILIA

    cudaSetDevice(0);

    const int num_thread_blocks = (std::max<int>(num_segs[0], num_blobs[0]) + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    #if ROD
      cufcm_solver->reform_xsegblob(x_segs_device[0], x_blobs_device[0], true);
      cufcm_solver->reform_fseg(f_segs_device[0], true);
      cufcm_solver->apply_repulsion();
      cufcm_solver->reform_fseg(f_segs_device[0], false);
      cufcm_solver->reform_fblob(f_blobs_repulsion_device[0], false);
    #endif

  #endif

}

void fcm_mobility_solver::wait_for_device(){

  for (int n = 0; n < num_gpus; n++){

      cudaSetDevice(n);
      cudaDeviceSynchronize();

    }

}

void fcm_mobility_solver::evaluate_segment_segment_mobility(){

  cudaSetDevice(0);
  cudaError_t err = cudaGetLastError();
  
  cufcm_solver->reform_xsegblob(x_segs_device[0], x_blobs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_xsegblob: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_fseg(f_segs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_fseg: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_fblob(f_blobs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_fblob: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->evaluate_mobility_cilia();
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error evaluate_mobility_cilia: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_vseg(v_segs_device[0], false);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_vseg: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_vblob(v_blobs_device[0], false);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_vblob: %s\n", cudaGetErrorString(err));
    }

  // cufcm_solver->write_data_call();
  
   // ########################### 
  // copy_segment_velocities_to_host();
  // copy_blob_velocities_to_host();
  // wait_for_device();

  // FILE *pfile;
  // pfile = fopen("sim_data_fcm.dat", "a");
  // for(int i = 0; i < NBLOB; i++){
  //   fprintf(pfile, "%d %.16f %.16f %.16f \n", 
  //   i, v_blobs_host[3*i + 0], v_blobs_host[3*i + 1], v_blobs_host[3*i + 2]);
  // }
  // for(int i = 0; i < NFIL; i++){
  //   fprintf(pfile, "%d %.16f %.16f %.16f \n", 
  //   i, v_segs_host[6*i + 0], v_segs_host[6*i + 1], v_segs_host[6*i + 2]);
  // }
  // fprintf(pfile, "\n#");
  // fclose(pfile);

  // // ########################### 
  // int start_seg = 0;
  // for (int n = 0; n < num_gpus; n++){
  //   cudaSetDevice(n);
  //   int num_thread_blocks = (num_segs[n] + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  //   Mss_mult<<<num_thread_blocks, THREADS_PER_BLOCK>>>(v_segs_device[n], f_segs_device[n], x_segs_device[n], start_seg, num_segs[n]);
  //   start_seg += num_segs[n];
  // }

  // copy_segment_velocities_to_host();
  // copy_blob_velocities_to_host();
  // wait_for_device();
  // pfile = fopen("sim_data_rpy.dat", "a");
  // for(int i = 0; i < NBLOB; i++){
  //   fprintf(pfile, "%d %.16f %.16f %.16f \n", 
  //   i, v_blobs_host[3*i + 0], v_blobs_host[3*i + 1], v_blobs_host[3*i + 2]);
  // }
  // for(int i = 0; i < NFIL; i++){
  //   fprintf(pfile, "%d %.16f %.16f %.16f \n", 
  //   i, v_segs_host[6*i + 0], v_segs_host[6*i + 1], v_segs_host[6*i + 2]);
  // }
  // fprintf(pfile, "\n#");
  // fclose(pfile);

  
}

void fcm_mobility_solver::evaluate_blob_blob_mobility(){
  cudaSetDevice(0);
  cudaError_t err = cudaGetLastError();
  
  cufcm_solver->reform_xsegblob(x_segs_device[0], x_blobs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_xsegblob: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_fseg(f_segs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_fseg: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_fblob(f_blobs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_fblob: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->evaluate_mobility_cilia();
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error evaluate_mobility_cilia: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_vseg(v_segs_device[0], false);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_vseg: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_vblob(v_blobs_device[0], false);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_vblob: %s\n", cudaGetErrorString(err));
    }

  // cufcm_solver->write_data_call();
  
   // ########################### 
  // copy_segment_velocities_to_host();
  // copy_blob_velocities_to_host();
  // wait_for_device();

  // FILE *pfile;
  // pfile = fopen("sim_blob_data_fcm.dat", "a");
  // for(int i = 0; i < NBLOB; i++){
  //     fprintf(pfile, "%d %.16f %.16f %.16f \n", 
  //     i, v_blobs_host[3*i + 0], v_blobs_host[3*i + 1], v_blobs_host[3*i + 2]);
  //     }
  // fprintf(pfile, "\n#");
  // fclose(pfile);

  // ########################### 
  // int start_blob = 0;
  // for (int n = 0; n < num_gpus; n++){
  //   cudaSetDevice(n);
  //   int num_thread_blocks = (num_blobs[n] + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  //   Mbb_mult<<<num_thread_blocks, THREADS_PER_BLOCK>>>(v_blobs_device[n], f_blobs_device[n], x_blobs_device[n], start_blob, num_blobs[n]);
  //   start_blob += num_blobs[n];
  // }

  // copy_segment_velocities_to_host();
  // copy_blob_velocities_to_host();
  // wait_for_device();
  // pfile = fopen("sim_blob_data_rpy.dat", "a");
  // for(int i = 0; i < NBLOB; i++){
  //     fprintf(pfile, "%d %.16f %.16f %.16f \n", 
  //     i, v_blobs_host[3*i + 0], v_blobs_host[3*i + 1], v_blobs_host[3*i + 2]);
  //     }
  // fprintf(pfile, "\n#");
  // fclose(pfile);

  
  
}

void fcm_mobility_solver::evaluate_segment_blob_mobility(){

  cudaSetDevice(0);
  int num_thread_blocks = (num_segs[0] + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

}

void fcm_mobility_solver::evaluate_blob_segment_mobility(){

  cudaSetDevice(0);
  int num_thread_blocks = (num_blobs[0] + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

}

void fcm_mobility_solver::write_repulsion(){
  FILE *pfile;
  cudaMemcpy(&f_blobs_repulsion_host[0], f_blobs_repulsion_device[0], 3*num_blobs[0]*sizeof(Real), cudaMemcpyDeviceToHost);
  pfile = fopen("barrier_force_fil.dat", "w");
  for(int i = 0; i < num_blobs[0]; i++){
      fprintf(pfile, "FIL %d %.8f %.8f %.8f %.8f %.8f %.8f \n", 
      i, f_blobs_repulsion_host[3*i], f_blobs_repulsion_host[3*i+1], f_blobs_repulsion_host[3*i+2],
      x_blobs_host[3*i], x_blobs_host[3*i+1], x_blobs_host[3*i+2]);
      }
  fprintf(pfile, "\n#");
  fclose(pfile);
}

void fcm_mobility_solver::copy_to_fcm(){
  // cufcm_solver->reform_xsegblob(x_segs_device[0], x_blobs_device[0], true);
  // cufcm_solver->reform_fblob(f_blobs_device[0], true);
  // cufcm_solver->reform_fseg(f_segs_device[0], true);
  // cufcm_solver->reform_vblob(v_blobs_device[0], true);
  // cufcm_solver->reform_vseg(v_segs_device[0], true);
}

void fcm_mobility_solver::evaluate_full_mobility(){
  cudaSetDevice(0);
  cudaError_t err = cudaGetLastError();
  
  cufcm_solver->reform_xsegblob(x_segs_device[0], x_blobs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_xsegblob: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_fseg(f_segs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_fseg: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_fblob(f_blobs_device[0], true);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_fblob: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->evaluate_mobility_cilia();
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error evaluate_mobility_cilia: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_vseg(v_segs_device[0], false);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_vseg: %s\n", cudaGetErrorString(err));
    }
  cufcm_solver->reform_vblob(v_blobs_device[0], false);
  err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reform_vblob: %s\n", cudaGetErrorString(err));
    }

  // cufcm_solver->write_data_call();
  
  copy_segment_velocities_to_host();
  copy_blob_velocities_to_host();
}