// seeding.cu

#include "seeding.hpp"
#include "../../config.hpp"
// #include "../../globals.hpp"

#if SURFACE_OF_REVOLUTION_BODIES or ROD or RIGIDWALL

  #include <iostream>
  #include <fstream>
  #include <cmath>
  #include <random>
  #include <string>
  #include "../general/util.hpp"
  #include "../general/matrix.hpp"

  __global__ void find_nearest_neighbours(int *const ids, const Real *const samples, const int num_samples, const Real *const X, const int N){

    // Work out which sample(s) this thread will compute the nearest-neighbour for
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    // Declare the shared memory for this thread block
    __shared__ Real x_shared[THREADS_PER_BLOCK];
    __shared__ Real y_shared[THREADS_PER_BLOCK];
    __shared__ Real z_shared[THREADS_PER_BLOCK];

    Real xi, yi, zi;
    int min_id;
    Real min_dist = 1e100;

    for (int i = index; (i-threadIdx.x) < num_samples; i+=stride){

      if (i < num_samples){

        xi = samples[3*i];
        yi = samples[3*i + 1];
        zi = samples[3*i + 2];

      }

      for (int j_start = 0; j_start < N; j_start += THREADS_PER_BLOCK){

        const int j_to_read = j_start + threadIdx.x;

        if (j_to_read < N){

          x_shared[threadIdx.x] = X[3*j_to_read];
          y_shared[threadIdx.x] = X[3*j_to_read + 1];
          z_shared[threadIdx.x] = X[3*j_to_read + 2];

        }

        __syncthreads();

        if (i < num_samples){

          for (int j=0; (j < THREADS_PER_BLOCK) && (j_start + j < N); j++){

            const Real dx = xi - x_shared[j];
            const Real dy = yi - y_shared[j];
            const Real dz = zi - z_shared[j];

            const Real dist = sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < min_dist){

              min_dist = dist;
              min_id = j_start + j;

            }

          }

        }

        __syncthreads();

      }

      if (i < num_samples){

        ids[i] = min_id;

      }

    }

  } // End of kernel.

  class shape_fourier_description{

  public:

  int num_modes;
  matrix cos_coeffs;
  matrix sin_coeffs;

  std::mt19937 gen;

  matrix polar_angle_area_fractions;

  ~shape_fourier_description(){};

  shape_fourier_description(){

    std::ifstream input_file(GENERATRIX_FILE_NAME+std::string(".fourier_modes"));

    if (input_file.good()){

      input_file >> num_modes;

      cos_coeffs = matrix(1, num_modes);
      sin_coeffs = matrix(1, num_modes);

      for (int n = 0; n < num_modes; n++){

        input_file >> cos_coeffs(n);
        input_file >> sin_coeffs(n);

      }

    } else {

      std::cout << std::endl << std::endl << " The required generatrix file '" << GENERATRIX_FILE_NAME << ".fourier_modes" << "' was not found." << std::endl;
      exit(-1);

    }

    input_file.close();

    if (std::string(GENERATRIX_FILE_NAME) != std::string("sphere")){

      input_file.open(GENERATRIX_FILE_NAME + std::string(".polar_angle_area_fractions"));

      if (input_file.good()){

        int num_theta_locs;

        input_file >> num_theta_locs;

        polar_angle_area_fractions = matrix(1, num_theta_locs);

        for (int n = 0; n < num_theta_locs; n++){

          input_file >> polar_angle_area_fractions(n);

        }

      } else {

        find_polar_angle_area_fractions();

      }

      input_file.close();

    }

    // Random seed for the RNG.
    std::random_device rd{};
    gen = std::mt19937{rd()};

  };

  Real radius(const Real theta) const {

    matrix cos_mat(num_modes,1);
    matrix sin_mat(num_modes,1);

    for (int n = 0; n < num_modes; n++){

      cos_mat(n) = std::cos(n*theta);
      sin_mat(n) = std::sin(n*theta);

    }

    return cos_coeffs*cos_mat + sin_coeffs*sin_mat;

  };

  matrix location(const Real theta, const Real phi) const {

    matrix loc(3,1);

    const Real r = radius(theta);

    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);

    const Real ct = std::cos(theta);
    const Real st = std::sin(theta);

    loc(0) = r*cp*st;
    loc(1) = r*sp*st;
    loc(2) = r*ct;

    return loc;

  };

  matrix azi_dir(const Real phi) const {

    matrix e(3,1);

    e(0) = -std::sin(phi);
    e(1) = std::cos(phi);
    e(2) = 0.0;

    return e;

  };

  Real radius_deriv(const Real theta) const {

    Real out = 0.0;

    for (int n = 1; n < num_modes; n++){

        out += n*(sin_coeffs(n)*std::cos(n*theta) - cos_coeffs(n)*std::sin(n*theta));

    }

    return out;

  };

  Real radius_second_deriv(const Real theta) const {

    Real out = 0.0;

    for (int n = 1; n < num_modes; n++){

        out -= n*n*(sin_coeffs(n)*std::sin(n*theta) + cos_coeffs(n)*std::cos(n*theta));

    }

    return out;

  };

  matrix polar_dir(const Real theta, const Real phi) const {

    const Real r = radius(theta);
    const Real rprime = radius_deriv(theta);

    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);

    const Real ct = std::cos(theta);
    const Real st = std::sin(theta);

    matrix e(3,1);

    const Real temp = rprime*st + r*ct;
    e(0) = cp*temp;
    e(1) = sp*temp;
    e(2) = rprime*ct - r*st;

    return e/norm(e);

  };

  matrix surface_normal(const Real theta, const Real phi) const {

    return cross(polar_dir(theta, phi), azi_dir(phi));

  };

  matrix full_frame(const Real theta, const Real phi) const {

    matrix frame(9,1);

    frame.set_block(0, 3, polar_dir(theta, phi));
    frame.set_block(3, 3, azi_dir(phi));
    frame.set_block(6, 3, cross(frame.get_block(0,3), frame.get_block(3,3)));

    return frame;

  };

  matrix full_frame_tilt(const Real theta, const Real phi, const Real tilt_angle) const {

    matrix frame(9,1);

    // frame.set_block(0, 3, cos(tilt_angle)*polar_dir(theta, phi) + sin(tilt_angle)*azi_dir(phi) );
    // frame.set_block(3, 3, -sin(tilt_angle)*polar_dir(theta, phi) + cos(tilt_angle)*azi_dir(phi) );
    frame.set_block(0, 3, azi_dir(phi));
    frame.set_block(3, 3, -polar_dir(theta, phi));
    frame.set_block(6, 3, cross(frame.get_block(0,3), frame.get_block(3,3)));

    return frame;

  };

  matrix random_point(){

    // N.B. The Mersenne twister "gen" is modified whenever we call upon it, so this method cannot be
    // const and consequently this class cannot be passed as const to the seeding function.

    matrix sample(3,1);

    if (std::string(GENERATRIX_FILE_NAME) == std::string(FOURIER_DIR) + std::string("sphere")){

      // There's a really simple way of generating uniform samples on a sphere.
      std::normal_distribution<Real> d(0.0, 1.0);

      sample(0) = d(gen);
      sample(1) = d(gen);
      sample(2) = d(gen);

      const Real sample_norm = norm(sample);

      sample /= sample_norm;

    } else {

      std::uniform_real_distribution<Real> d(0.0, 1.0);

      const Real u1 = d(gen);
      const Real phi = 2.0*PI*u1;

      const Real u2 = d(gen);

      const int num_theta = polar_angle_area_fractions.num_cols * polar_angle_area_fractions.num_rows; // So it doesn't matter whether I made it a row vector or a column vector.

      for (int n = 1; n < num_theta; n++){

        if (u2 < polar_angle_area_fractions(n)){

          const Real frac = (u2 - polar_angle_area_fractions(n-1))/(polar_angle_area_fractions(n) - polar_angle_area_fractions(n-1));
          const Real theta_l = PI*(n-1)/Real(num_theta-1);
          const Real dtheta = PI/Real(num_theta-1);

          const Real theta = theta_l + frac*dtheta;

          sample = location(theta, phi);

          break;

        }

      }

    }

    return sample;

  };

  matrix random_point_away_from_poles(const Real shift_ratio){

    // N.B. The Mersenne twister "gen" is modified whenever we call upon it, so this method cannot be
    // const and consequently this class cannot be passed as const to the seeding function.

    matrix sample(3,1);

    if (std::string(GENERATRIX_FILE_NAME) != std::string(FOURIER_DIR) + std::string("sphere")){

      std::cout<< "WARNING: Non-spherical shape not implemented." << std::endl;

    }

      // There's a really simple way of generating uniform samples on a sphere.
      std::normal_distribution<Real> d(0.0, 1.0);

      sample(0) = d(gen);
      sample(1) = d(gen);
      sample(2) = d(gen);

      const Real sample_norm = norm(sample);
      
      sample /= sample_norm;

      Real theta = std::atan2(std::sqrt(sample(0)*sample(0) + sample(1)*sample(1)),sample(2));
      const Real phi = std::atan2(sample(1), sample(0));

      theta = theta*(1-2*shift_ratio)+shift_ratio*PI;

      sample(0) = sample_norm*std::sin(theta)*std::sin(phi);
      sample(1) = sample_norm*std::sin(theta)*std::cos(phi);
      sample(2) = sample_norm*std::cos(theta);
     
      sample /= sample_norm;

    return sample;

  };

  // matrix random_point_away_from_poles_and_inhomogeneous(const Real shift_ratio, const Real factor){

  //   // N.B. The Mersenne twister "gen" is modified whenever we call upon it, so this method cannot be
  //   // const and consequently this class cannot be passed as const to the seeding function.

  //   matrix sample(3,1);

  //   if (std::string(GENERATRIX_FILE_NAME) == std::string(FOURIER_DIR) + std::string("sphere")){

  //     // There's a really simple way of generating uniform samples on a sphere.
  //     std::normal_distribution<Real> d(0.0, 1.0);

  //     sample(0) = d(gen)*factor;
  //     sample(1) = d(gen)*factor;
  //     sample(2) = d(gen);

  //     const Real sample_norm = norm(sample);
      
  //     sample /= sample_norm;

  //     Real theta = std::atan2(std::sqrt(sample(0)*sample(0) + sample(1)*sample(1)),sample(2));
  //     const Real phi = std::atan2(sample(1), sample(0));

  //     theta = theta*(1-2*shift_ratio)+shift_ratio*PI;

  //     sample(0) = sample_norm*std::sin(theta)*std::sin(phi);
  //     sample(1) = sample_norm*std::sin(theta)*std::cos(phi);
  //     sample(2) = sample_norm*std::cos(theta);
     
  //     sample /= sample_norm;

  //   } else {

  //     std::cout<< "Non-spherical shape not implemented." << std::endl;

  //   }

  //   return sample;

  // };

  Real polar_angle_area_fraction_integrand(const Real theta) const {

    const Real r = radius(theta);
    const Real rprime = radius_deriv(theta);

    return r*std::sin(theta)*std::sqrt(r*r + rprime*rprime);

  };

  void find_polar_angle_area_fractions(){

    int num_theta_locs = 1000;

    polar_angle_area_fractions = matrix(1, num_theta_locs);

    polar_angle_area_fractions(0) = 0.0;

    Real F_old = 0.0;

    for (int n = 1; n < num_theta_locs; n++){

      const Real theta = PI*n/Real(num_theta_locs-1);
      const Real F = polar_angle_area_fraction_integrand(theta);
      polar_angle_area_fractions(n) = polar_angle_area_fractions(n-1) + 0.5*PI*(F + F_old)/Real(num_theta_locs-1); // Trapezium rule.

      F_old = F;

    }

    // Scale by the total (which is actually the total area of the shape divided by 2pi)
    for (int n = 1; n < num_theta_locs-1; n++){

      polar_angle_area_fractions(n) /= polar_angle_area_fractions(num_theta_locs-1);

    }

    polar_angle_area_fractions(num_theta_locs-1) = 1.0; // Do this to avoid any rounding-error fishyness.

    // Write to file
    std::ofstream file(GENERATRIX_FILE_NAME+std::string(".polar_angle_area_fractions"));

    file << num_theta_locs << " ";

    for (int n = 0; n < num_theta_locs; n++){

      file << polar_angle_area_fractions(n) << " ";

    }

    file.close();

  };

  Real surface_projection_error(const Real theta, const Real x1, const Real x2) const {

    const Real r = radius(theta);
    const Real rp = radius_deriv(theta);

    const Real c = std::cos(theta);
    const Real s = std::sin(theta);

    return r*rp + x1*(r*s - rp*c) - x2*(r*c + rp*s);

  };

  Real surface_projection_error_deriv(const Real theta, const Real x1, const Real x2) const {

    const Real r = radius(theta);
    const Real rp = radius_deriv(theta);
    const Real rpp = radius_second_deriv(theta);

    const Real c = std::cos(theta);
    const Real s = std::sin(theta);

    return r*rpp + rp*rp + x1*((r-rpp)*c + 2.0*rp*s) - x2*((rpp-r)*s + 2.0*rp*c);

  };

  void project_onto_surface(Real *const X) const {

    Real theta = std::atan2(std::sqrt(X[0]*X[0] + X[1]*X[1]), X[2]);

    if (std::string(GENERATRIX_FILE_NAME) == std::string("sphere")){

      const Real curr_radius = std::sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
      const Real targ_radius = radius(theta);

      X[0] *= targ_radius/curr_radius;
      X[1] *= targ_radius/curr_radius;
      X[2] *= targ_radius/curr_radius;

    } else {

      const Real x1 = X[2];
      const Real x2 = std::sqrt(X[0]*X[0] + X[1]*X[1]);

      Real error = surface_projection_error(theta, x1, x2);

      while (error > 1e-2){

        const Real error_prime = surface_projection_error_deriv(theta, x1, x2);

        theta -= error/error_prime; // Newton's method.

        error = surface_projection_error(theta, x1, x2);

      }

      const Real phi = std::atan2(X[1], X[0]);

      const matrix loc = location(theta, phi);

      X[0] = loc(0);
      X[1] = loc(1);
      X[2] = loc(2);

    }

  };

  };







  void equal_area_seeding(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape){

    if (N == 0){

      return; // Because of the squirmer-style simulations, the code may try to seed 0 filaments.

    }

    // This function essentially implements a version of MacQueen's algorithm.
    // We seed a large number of points randomly on the surface according to a uniform distribution, and then find
    // our N points as the centres of regions of (approximately) equal area through N-means clustering.

    int samples_per_iter = 1000;

    // Allocate memory for the CUDA nearest-neighbour search.
    Real *samples, *X;
    cudaMallocManaged(&samples, 3*samples_per_iter*sizeof(Real));
    cudaMallocManaged(&X, 3*N*sizeof(Real));

    int *sample_nn_ids;
    cudaMallocManaged(&sample_nn_ids, samples_per_iter*sizeof(int));

    const int num_thread_blocks = (samples_per_iter + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // Generate initial guesses.
    if (std::string(GENERATRIX_FILE_NAME) == std::string("sphere")){

      // We use the spiral distribution given by Saff and Kuijlaars (1997) as our initial positions.
      Real phi = 0.0;

      for (int n = 0; n < N; n++){

        const Real theta = std::acos((N == 1) ? -1.0 : 2.0*n/Real(N-1) - 1.0);

        const matrix d = shape.location(theta, phi);

        X[3*n] = d(0);
        X[3*n + 1] = d(1);
        X[3*n + 2] = d(2);

        if (n == N-2){

          phi = 0.0;

        } else {

          const Real r = std::sqrt(d(0)*d(0) + d(1)*d(1));

          phi += 3.6/(r*std::sqrt(N));

        }

      }

    } else {

      // Use random initial positions.
      for (int n = 0; n < N; n++){

        const matrix sample = shape.random_point();
        X[3*n] = sample(0);
        X[3*n + 1] = sample(1);
        X[3*n + 2] = sample(2);

      }

    }

    std::vector<int> count(N);
    std::vector<int> old_count(N); // Used to check for convergence.

    for (int n = 0; n < N; n++){

      count[n] = 1;
      old_count[n] = 1;

    }

    int num_iters = 0;

    while (num_iters < 10000){

      // Sample from the uniform distribution on the surface.
      for (int n = 0; n < samples_per_iter; n++){

        const matrix sample = shape.random_point();

        samples[3*n] = sample(0);
        samples[3*n + 1] = sample(1);
        samples[3*n + 2] = sample(2);

      }
      

      // Find the candidate points closest to the random samples.
      find_nearest_neighbours<<<num_thread_blocks, THREADS_PER_BLOCK>>>(sample_nn_ids, samples, samples_per_iter, X, N);
      cudaDeviceSynchronize();

      // Update the candidate points.
      for (int n = 0; n < samples_per_iter; n++){

        const int min_id = sample_nn_ids[n];

        X[3*min_id] = (count[min_id]*X[3*min_id] + samples[3*n])/Real(count[min_id] + 1);
        X[3*min_id + 1] = (count[min_id]*X[3*min_id + 1] + samples[3*n + 1])/Real(count[min_id] + 1);
        X[3*min_id + 2] = (count[min_id]*X[3*min_id + 2] + samples[3*n + 2])/Real(count[min_id] + 1);

        count[min_id]++;

        shape.project_onto_surface(&X[3*min_id]);

      }

      num_iters++;

      // Check for convergence.
      // In general, I only expect this to trigger an early exit for small numbers of points (e.g. some filament seeding problems),
      // and even then possibly only on spheres (or any other shapes that actually have some exact solutions).
      if (num_iters % 100 == 0){

        int max_change = 0;
        int min_change = 1000000;

        for (int n = 0; n < N; n++){

          if (max_change < count[n]-old_count[n]){

            max_change = count[n] - old_count[n];

          }

          if (min_change > count[n]-old_count[n]){

            min_change = count[n] - old_count[n];

          }

        }

        old_count = count;

        if (Real(min_change)/Real(max_change) > 0.99){

          break;

        }

      }

    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = X[3*n];
      pos_ref[3*n + 1] = X[3*n + 1];
      pos_ref[3*n + 2] = X[3*n + 2];

      const Real theta = std::atan2(std::sqrt(X[3*n]*X[3*n] + X[3*n + 1]*X[3*n + 1]), X[3*n + 2]);
      const Real phi = std::atan2(X[3*n + 1], X[3*n]);

      matrix frame = shape.full_frame(theta, phi);

      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }

  };

  void equal_area_seeding_poles(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape){

    if (N == 0){

      return; // Because of the squirmer-style simulations, the code may try to seed 0 filaments.

    }

    // This function essentially implements a version of MacQueen's algorithm.
    // We seed a large number of points randomly on the surface according to a uniform distribution, and then find
    // our N points as the centres of regions of (approximately) equal area through N-means clustering.

    int samples_per_iter = 1000;

    // Real shift_ratio = (0.471*FIL_LENGTH)/(0.5*AXIS_DIR_BODY_LENGTH*PI); // value used from Dec 2023 to Feb 2024
    Real shift_ratio = (1.0*FIL_LENGTH)/(0.5*AXIS_DIR_BODY_LENGTH*PI); // value used from Feb 2024
    // Real shift_ratio = (0.2*FIL_LENGTH)/(0.5*AXIS_DIR_BODY_LENGTH*PI); 
    int max_num_iters = 10000;

    // debug begins
    // std::ifstream in("input/shift_ratio/shift_ratio.dat"); // input
    //       in >> shift_ratio;
    //       in >> max_num_iters;
    //       in >> samples_per_iter;
    //       in.close();
    //debug ends

    // Allocate memory for the CUDA nearest-neighbour search.
    Real *samples, *X;
    cudaMallocManaged(&samples, 3*samples_per_iter*sizeof(Real));
    cudaMallocManaged(&X, 3*N*sizeof(Real));

    int *sample_nn_ids;
    cudaMallocManaged(&sample_nn_ids, samples_per_iter*sizeof(int));

    const int num_thread_blocks = (samples_per_iter + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // Generate initial guesses.
    if (std::string(GENERATRIX_FILE_NAME) == std::string("sphere")){

      // We use the spiral distribution given by Saff and Kuijlaars (1997) as our initial positions.
      Real phi = 0.0;

      for (int n = 0; n < N; n++){

        const Real theta = std::acos((N == 1) ? -1.0 : 2.0*n/Real(N-1) - 1.0);

        const matrix d = shape.location(theta, phi);

        X[3*n] = d(0);
        X[3*n + 1] = d(1);
        X[3*n + 2] = d(2);

        if (n == N-2){

          phi = 0.0;

        } else {

          const Real r = std::sqrt(d(0)*d(0) + d(1)*d(1));

          phi += 3.6/(r*std::sqrt(N));

        }

      }

    } else {

      // Use random initial positions.
      for (int n = 0; n < N; n++){

        // const matrix sample = shape.random_point();
        const matrix sample = shape.random_point_away_from_poles(shift_ratio);
        X[3*n] = sample(0);
        X[3*n + 1] = sample(1);
        X[3*n + 2] = sample(2);

      }

    }

    std::vector<int> count(N);
    std::vector<int> old_count(N); // Used to check for convergence.

    for (int n = 0; n < N; n++){

      count[n] = 1;
      old_count[n] = 1;

    }

    int num_iters = 0;


    while (num_iters < max_num_iters){

      // Sample from the uniform distribution on the surface.
      for (int n = 0; n < samples_per_iter; n++){

        const matrix sample = shape.random_point_away_from_poles(shift_ratio);

        samples[3*n] = sample(0);
        samples[3*n + 1] = sample(1);
        samples[3*n + 2] = sample(2);

      }

      // Find the candidate points closest to the random samples.
      find_nearest_neighbours<<<num_thread_blocks, THREADS_PER_BLOCK>>>(sample_nn_ids, samples, samples_per_iter, X, N);
      cudaDeviceSynchronize();

      // Update the candidate points.
      for (int n = 0; n < samples_per_iter; n++){

        const int min_id = sample_nn_ids[n];

        // Hang: min_id == -1 means the nearest neighbour is the aux-point which we want filaments to avoid.
        if(!(min_id == -1)){
          X[3*min_id] = (count[min_id]*X[3*min_id] + samples[3*n])/Real(count[min_id] + 1);
          X[3*min_id + 1] = (count[min_id]*X[3*min_id + 1] + samples[3*n + 1])/Real(count[min_id] + 1);
          X[3*min_id + 2] = (count[min_id]*X[3*min_id + 2] + samples[3*n + 2])/Real(count[min_id] + 1);

          count[min_id]++;

          shape.project_onto_surface(&X[3*min_id]);
        }
      }

      num_iters++;

      // Check for convergence.
      // In general, I only expect this to trigger an early exit for small numbers of points (e.g. some filament seeding problems),
      // and even then possibly only on spheres (or any other shapes that actually have some exact solutions).
      if (num_iters % 100 == 0){

        int max_change = 0;
        int min_change = 1000000;

        for (int n = 0; n < N; n++){

          if (max_change < count[n]-old_count[n]){

            max_change = count[n] - old_count[n];

          }

          if (min_change > count[n]-old_count[n]){

            min_change = count[n] - old_count[n];

          }

        }

        old_count = count;

        if (Real(min_change)/Real(max_change) > 0.99){

          break;

        }

      }

    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = X[3*n];
      pos_ref[3*n + 1] = X[3*n + 1];
      pos_ref[3*n + 2] = X[3*n + 2];

      const Real theta = std::atan2(std::sqrt(X[3*n]*X[3*n] + X[3*n + 1]*X[3*n + 1]), X[3*n + 2]);
      const Real phi = std::atan2(X[3*n + 1], X[3*n]);

      matrix frame = shape.full_frame(theta, phi);
      // N.B. the polar and azi refs are not used at the end
      // To define the beating plane, consult the filament class instead
      
      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }

  };

  void equal_area_seeding_poles_pair(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape){

    if (N == 0){

      return; // Because of the squirmer-style simulations, the code may try to seed 0 filaments.

    }

    int NP = N/2;

    // This function essentially implements a version of MacQueen's algorithm.
    // We seed a large number of points randomly on the surface according to a uniform distribution, and then find
    // our N points as the centres of regions of (approximately) equal area through N-means clustering.

    int samples_per_iter = 1000;

    // Real shift_ratio = (0.471*FIL_LENGTH)/(0.5*AXIS_DIR_BODY_LENGTH*PI); // value used from Dec 2023 to Feb 2024
    Real shift_ratio = (1.0*FIL_LENGTH)/(0.5*AXIS_DIR_BODY_LENGTH*PI); // value used from Feb 2024
    int max_num_iters = 10000;

    // Allocate memory for the CUDA nearest-neighbour search.
    Real *samples, *X;
    cudaMallocManaged(&samples, 3*samples_per_iter*sizeof(Real));
    cudaMallocManaged(&X, 3*NP*sizeof(Real));

    int *sample_nn_ids;
    cudaMallocManaged(&sample_nn_ids, samples_per_iter*sizeof(int));

    const int num_thread_blocks = (samples_per_iter + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // Generate initial guesses.
    if (std::string(GENERATRIX_FILE_NAME) == std::string("sphere")){

      // We use the spiral distribution given by Saff and Kuijlaars (1997) as our initial positions.
      Real phi = 0.0;

      for (int n = 0; n < NP; n++){

        const Real theta = std::acos((NP == 1) ? -1.0 : 2.0*n/Real(NP-1) - 1.0);

        const matrix d = shape.location(theta, phi);

        X[3*n] = d(0);
        X[3*n + 1] = d(1);
        X[3*n + 2] = d(2);

        if (n == NP-2){

          phi = 0.0;

        } else {

          const Real r = std::sqrt(d(0)*d(0) + d(1)*d(1));

          phi += 3.6/(r*std::sqrt(NP));

        }

      }

    } else {

      // Use random initial positions.
      for (int n = 0; n < NP; n++){

        // const matrix sample = shape.random_point();
        const matrix sample = shape.random_point_away_from_poles(shift_ratio);
        X[3*n] = sample(0);
        X[3*n + 1] = sample(1);
        X[3*n + 2] = sample(2);

      }

    }

    std::vector<int> count(NP);
    std::vector<int> old_count(NP); // Used to check for convergence.

    for (int n = 0; n < NP; n++){

      count[n] = 1;
      old_count[n] = 1;

    }

    int num_iters = 0;


    while (num_iters < max_num_iters){

      // Sample from the uniform distribution on the surface.
      for (int n = 0; n < samples_per_iter; n++){

        const matrix sample = shape.random_point_away_from_poles(shift_ratio);

        samples[3*n] = sample(0);
        samples[3*n + 1] = sample(1);
        samples[3*n + 2] = sample(2);

      }

      // Find the candidate points closest to the random samples.
      find_nearest_neighbours<<<num_thread_blocks, THREADS_PER_BLOCK>>>(sample_nn_ids, samples, samples_per_iter, X, NP);
      cudaDeviceSynchronize();

      // Update the candidate points.
      for (int n = 0; n < samples_per_iter; n++){

        const int min_id = sample_nn_ids[n];

        // Hang: min_id == -1 means the nearest neighbour is the aux-point which we want filaments to avoid.
        if(!(min_id == -1)){
          X[3*min_id] = (count[min_id]*X[3*min_id] + samples[3*n])/Real(count[min_id] + 1);
          X[3*min_id + 1] = (count[min_id]*X[3*min_id + 1] + samples[3*n + 1])/Real(count[min_id] + 1);
          X[3*min_id + 2] = (count[min_id]*X[3*min_id + 2] + samples[3*n + 2])/Real(count[min_id] + 1);

          count[min_id]++;

          shape.project_onto_surface(&X[3*min_id]);
        }
      }

      num_iters++;

      // Check for convergence.
      // In general, I only expect this to trigger an early exit for small numbers of points (e.g. some filament seeding problems),
      // and even then possibly only on spheres (or any other shapes that actually have some exact solutions).
      if (num_iters % 100 == 0){

        int max_change = 0;
        int min_change = 1000000;

        for (int n = 0; n < NP; n++){

          if (max_change < count[n]-old_count[n]){

            max_change = count[n] - old_count[n];

          }

          if (min_change > count[n]-old_count[n]){

            min_change = count[n] - old_count[n];

          }

        }

        old_count = count;

        if (Real(min_change)/Real(max_change) > 0.99){

          break;

        }

      }

    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      int pair_index = 3*(n%NP);

      std::cout << n << " " << pair_index << " " << std::endl;

      Real x, y, z;

      // flagellum 1 in the pair
      if (n/NP == 0){
        x = X[pair_index];
        y = X[pair_index + 1];
        z = X[pair_index + 2];
      }
      // flagellum 2 in the pair
      else{
        Real phi = atan2(X[pair_index + 1], X[pair_index]);
        Real theta = acos(X[pair_index + 2]/(sqrt(X[pair_index]*X[pair_index]+
                                                  X[pair_index + 1]*X[pair_index + 1]+
                                                  X[pair_index + 2]*X[pair_index + 2])));
        Real rotation_angle = 2*PI*(2.6*RSEG/(AXIS_DIR_BODY_LENGTH*PI*sin(theta)));
        x = cos(rotation_angle)*X[pair_index] - sin(rotation_angle)*X[pair_index + 1];
        y = sin(rotation_angle)*X[pair_index] + cos(rotation_angle)*X[pair_index + 1];
        z = X[pair_index + 2];
      }

      pos_ref[3*n] = x;
      pos_ref[3*n + 1] = y;
      pos_ref[3*n + 2] = z;

      
      

      const Real theta = std::atan2(std::sqrt(x*x + y*y), z);
      const Real phi = std::atan2(y, x);

      matrix frame = shape.full_frame(theta, phi);
      // N.B. the polar and azi refs are not used at the end
      // To define the beating plane, consult the filament class instead
      
      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }

  };

  std::mt19937 gen1;

  matrix random_point_on_a_disc(float singularity_shift){
    
    matrix sample(3,1);
    // There's a really simple way of generating uniform samples on a sphere.
    Real sample_norm = 1e100;
    while(sample_norm > 1 || sample_norm < singularity_shift){
      std::uniform_real_distribution<> d(-1.0, 1.0);

      sample(0) = d(gen1);
      sample(1) = d(gen1);
      sample(2) = 0;

      sample_norm = norm(sample);
    }
    
    sample /= sample_norm;

    // Real theta = std::atan2(sample(0), sample(1));
    const Real phi = std::atan2(sample(1), sample(0));

    sample(0) = sample_norm*std::sin(phi);
    sample(1) = sample_norm*std::cos(phi);
    sample(2) = 0;
    
    // sample /= sample_norm;
    return sample;
  }

  void equal_area_seeding_centric(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape, Real disc_r, Real singularity){

    if (N == 0){

      return; // Because of the squirmer-style simulations, the code may try to seed 0 filaments.

    }

    // This function essentially implements a version of MacQueen's algorithm.
    // We seed a large number of points randomly on the surface according to a uniform distribution, and then find
    // our N points as the centres of regions of (approximately) equal area through N-means clustering.

    int samples_per_iter = 1000;
    
    // Allocate memory for the CUDA nearest-neighbour search.
    Real *samples, *X;
    cudaMallocManaged(&samples, 3*samples_per_iter*sizeof(Real));
    cudaMallocManaged(&X, 3*N*sizeof(Real));

    int *sample_nn_ids;
    cudaMallocManaged(&sample_nn_ids, samples_per_iter*sizeof(int));

    const int num_thread_blocks = (samples_per_iter + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // Use random initial positions.
    for (int n = 0; n < N; n++){

      const matrix sample = random_point_on_a_disc(singularity);
      X[3*n] = sample(0);
      X[3*n + 1] = sample(1);
      X[3*n + 2] = sample(2);

    }


    std::vector<int> count(N);
    std::vector<int> old_count(N); // Used to check for convergence.
    for (int n = 0; n < N; n++){

      count[n] = 1;
      old_count[n] = 1;

    }

    int num_iters = 0;

    while (num_iters < 10000){

      // Sample from the uniform distribution on the surface.
      for (int n = 0; n < samples_per_iter; n++){

        const matrix sample = random_point_on_a_disc(singularity);

        samples[3*n] = sample(0);
        samples[3*n + 1] = sample(1);
        samples[3*n + 2] = sample(2);

      }
      
      // Find the candidate points closest to the random samples.
      find_nearest_neighbours<<<num_thread_blocks, THREADS_PER_BLOCK>>>(sample_nn_ids, samples, samples_per_iter, X, N);
      cudaDeviceSynchronize();

      // Update the candidate points.
      for (int n = 0; n < samples_per_iter; n++){

        const int min_id = sample_nn_ids[n];

        X[3*min_id] = (count[min_id]*X[3*min_id] + samples[3*n])/Real(count[min_id] + 1);
        X[3*min_id + 1] = (count[min_id]*X[3*min_id + 1] + samples[3*n + 1])/Real(count[min_id] + 1);
        X[3*min_id + 2] = (count[min_id]*X[3*min_id + 2] + samples[3*n + 2])/Real(count[min_id] + 1);

        count[min_id]++;

      }

      num_iters++;

      // // Check for convergence.
      // // In general, I only expect this to trigger an early exit for small numbers of points (e.g. some filament seeding problems),
      // // and even then possibly only on spheres (or any other shapes that actually have some exact solutions).
      // if (num_iters % 100 == 0){

      //   int max_change = 0;
      //   int min_change = 1000000;

      //   for (int n = 0; n < N; n++){

      //     if (max_change < count[n]-old_count[n]){

      //       max_change = count[n] - old_count[n];

      //     }

      //     if (min_change > count[n]-old_count[n]){

      //       min_change = count[n] - old_count[n];

      //     }

      //   }

      //   old_count = count;

      //   if (Real(min_change)/Real(max_change) > 0.99){

      //     break;

      //   }

      // }

    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = disc_r*X[3*n];
      pos_ref[3*n + 1] = disc_r*X[3*n + 1];
      pos_ref[3*n + 2] = disc_r*X[3*n + 2];

      const Real theta = 0.001;
      // const Real theta = std::atan2(std::sqrt(X[3*n]*X[3*n] + X[3*n + 1]*X[3*n + 1]), X[3*n + 2]);
      // matrix sample(3,1);
      // sample(0)
      // const Real sample_norm = norm(sample);
      
      const Real phi = std::atan2(X[3*n + 1], X[3*n]);

      matrix frame = shape.full_frame(theta, phi);

      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }

  };

  void hexagonal_seeding(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape,
                         Real step_x, int dim_x, int hex_num, float rev_ratio){
    
    /*Implement the hexagonal seeding algorithm here.*/
    Real *X = (Real*) malloc(3*N*sizeof(Real));
    Real step = step_x;
    int grid_dim_x = dim_x;

    const int grid_dim_y = std::max<int>(1, int(ceil(N/Real(grid_dim_x))));
    const double grid_step_x = step;
    const double grid_step_y = step;

    for (int i = 0; i < grid_dim_x; i++){
      for (int j = 0; j < grid_dim_y; j++){

        const int blob_id = j + i*grid_dim_y;

        if (blob_id < N){

          X[3*blob_id + 0] = i*grid_step_x + 0.5*(1+(j%hex_num))*grid_step_x;
          X[3*blob_id + 1] = j*grid_step_y + 0.5*grid_step_y;
          X[3*blob_id + 2] = 0.0;
        }
      }
    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = X[3*n];
      pos_ref[3*n + 1] = X[3*n + 1];
      pos_ref[3*n + 2] = X[3*n + 2];
      
      const Real theta = 0.001;
      Real phi = 0.5*PI;
      if(pos_ref[3*n + 1] >= rev_ratio*grid_step_y*grid_dim_y){
        phi +=  PI;
      }

      matrix frame = shape.full_frame(theta, phi);

      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }
  }

  void mismatch_seeding(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape){
    
    /*Implement the mismatched algorithm here.*/
    Real R = 1;
    double theta0 = (0.05*PI);
    int N0 = 64;
    double dtheta = (0.023*PI);
    Real dx = 2*R*PI*sin(theta0)/N0;
    Real dy = R*sin(dtheta);

    int N_layer = ceil((0.5*PI-theta0)/dtheta);
    
    Real *theta_list = (Real*) malloc(N_layer*sizeof(Real));
    Real *N_list = (Real*) malloc(N_layer*sizeof(Real));
  
    for(int i=0; i<N_layer; i++){
      theta_list[i] = theta0;
      N_list[i] = N0;

      theta0 = theta0 + dtheta;
      N0 = int(2*PI*sin(theta0)/dx);
      
    }

    int num_fil = 0;
    for (int i = 0; i < N_layer; i++) {
        num_fil += N_list[i];
    }
    Real *X = (Real*) malloc(6*num_fil*sizeof(Real));
    int index = 0;
    Real phi0 = 0;
    Real dphi = 0;

    for(int i=0; i<N_layer; i++){
      phi0 += 0.5*dphi;
      dphi = 2*PI/N_list[i];
      for(int j=0; j<N_list[i]; j++){
        CartesianCoordinates cartesian1 = spherical_to_cartesian(R, theta_list[i], phi0+dphi*j);
        X[3*index] = 0.5*cartesian1.x;
        X[3*index+1] = 0.5*cartesian1.y;
        X[3*index+2] = 0.5*cartesian1.z;
        CartesianCoordinates cartesian2 = spherical_to_cartesian(R, PI-theta_list[i], phi0+dphi*(0.5+j));
        X[3*index+3*num_fil] = 0.5*cartesian2.x;
        X[3*index+1+3*num_fil] = 0.5*cartesian2.y;
        X[3*index+2+3*num_fil] = 0.5*cartesian2.z;
        index ++;
      }
    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = X[3*n];
      pos_ref[3*n + 1] = X[3*n + 1];
      pos_ref[3*n + 2] = X[3*n + 2];

      const Real theta = std::atan2(std::sqrt(X[3*n]*X[3*n] + X[3*n + 1]*X[3*n + 1]), X[3*n + 2]);
      const Real phi = std::atan2(X[3*n + 1], X[3*n]);

      matrix frame = shape.full_frame(theta, phi);

      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }
  }

  void rod_seeding(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape){
    
    /*Implement the mismatched algorithm here.*/
    Real *X = (Real*) malloc(3*NBLOB*sizeof(Real));
    int pair = 0;

    for (int n=0; n<NBLOB; n+=2){

      const Real theta = pair*0.5*PI;
      const Real x = sqrt(2.0)*RBLOB*(pair - 0.25*NBLOB + 0.5);

      X[3*n] = x;
      X[3*n + 1] = RBLOB*cos(theta);
      X[3*n + 2] = RBLOB*sin(theta);

      X[3*(n+1)] = x;
      X[3*(n+1) + 1] = -RBLOB*cos(theta);
      X[3*(n+1) + 2] = -RBLOB*sin(theta);

      pair++;

    }
    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = X[3*n];
      pos_ref[3*n + 1] = X[3*n + 1];
      pos_ref[3*n + 2] = X[3*n + 2];

      const Real theta = std::atan2(std::sqrt(X[3*n]*X[3*n] + X[3*n + 1]*X[3*n + 1]), X[3*n + 2]);
      const Real phi = std::atan2(X[3*n + 1], X[3*n]);

      matrix frame = shape.full_frame(theta, phi);

      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }
  }

  void icosa_seeding(Real *const pos_ref, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs, const int N, shape_fourier_description& shape, const std::string ico_file){
    
    Real *X = (Real*) malloc(3*N*sizeof(Real));

    std::ifstream inputFile(ico_file); 
    std::cout << std::endl << std::endl << "Using icosahedroal grid as seeding function..." << std::endl;
    std::cout << std::endl << std::endl << "Input file " << ico_file << std::endl;

    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file.\n";
    }

    for(int i=0; i<N; i++){
      inputFile >> X[3*i];
      inputFile >> X[3*i+1];
      inputFile >> X[3*i+2];
    }

    for(int i=0; i<N; i++){
      shape.project_onto_surface(&X[3*i]);
    }

    // Write the data for the final positions
    for (int n = 0; n < N; n++){

      pos_ref[3*n] = X[3*n];
      pos_ref[3*n + 1] = X[3*n + 1];
      pos_ref[3*n + 2] = X[3*n + 2];

      const Real theta = std::atan2(std::sqrt(X[3*n]*X[3*n] + X[3*n + 1]*X[3*n + 1]), X[3*n + 2]);
      const Real phi = std::atan2(X[3*n + 1], X[3*n]);

      matrix frame = shape.full_frame(theta, phi);

      polar_dir_refs[3*n] = frame(0);
      polar_dir_refs[3*n + 1] = frame(1);
      polar_dir_refs[3*n + 2] = frame(2);

      azi_dir_refs[3*n] = frame(3);
      azi_dir_refs[3*n + 1] = frame(4);
      azi_dir_refs[3*n + 2] = frame(5);

      normal_refs[3*n] = frame(6);
      normal_refs[3*n + 1] = frame(7);
      normal_refs[3*n + 2] = frame(8);

    }
  }

  void seed_blobs(Real *const blob_references, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs){

    const std::string file_name_trunk = GENERATRIX_FILE_NAME+std::to_string(NBLOB);

    std::cout << std::endl << std::endl << "Generating seeding files " << "'" << file_name_trunk << ".seed'" << std::endl;
    
    shape_fourier_description shape;

    #if ICOSA_SEEDING
      // std::cout << "Seeking an equal-area distribution for the blobs..." << std::endl;
      // equal_area_seeding(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape);
    
      std::cout << "Seeking an icosahedral distribution for the blobs..." << std::endl;
      const std::string ico_file = SIMULATION_BLOBPLACEMENT_NAME;
      icosa_seeding(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape, ico_file);
    #elif ROD
      std::cout << "Seeking an rod placement for the blobs..." << std::endl;
      rod_seeding(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape);
    #elif RIGIDWALL

      #if HEXAGONAL_WALL_SEEDING
        std::cout << "Seeking an hexagonal placement for the blobs..." << std::endl;
        // Real blob_step_x;
        // int blob_grid_dim_x;
        // int hex_num;
        // Real rev_ratio;
        // blob_step_x = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "blob_spacing"));
        // blob_grid_dim_x = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "blob_x_dim"));
        // hex_num = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "hex_num"));
        // rev_ratio = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "reverse_fil_direction_ratio"));
        // hexagonal_seeding(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape, blob_step_x, blob_grid_dim_x, hex_num, rev_ratio);
        hexagonal_seeding(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape, BLOB_SPACING, BLOB_X_DIM, HEX_NUM, REV_RATIO);
      
      #elif CENTRIC_WALL_SEEDING
        Real disc_radius = BLOB_SPACING*my_sqrt(NBLOB);
        // std::ifstream in("separation.dat"); // using the 6th number as input
        // for (int di=0; di<6; di++){
        //   in >> disc_radius;
        // }
        std::cout << "Seeking an centric placement for the blobs..." << std::endl;
        equal_area_seeding_centric(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape, disc_radius, 0.0);
      #endif
    #else
      std::cout << "Seeking an equal-area distribution for the blobs..." << std::endl;
      equal_area_seeding(blob_references, polar_dir_refs, azi_dir_refs, normal_refs, NBLOB, shape);
    #endif

    std::ofstream blob_ref_file(file_name_trunk + ".seed");
    std::ofstream polar_file(file_name_trunk + ".polar_dir");
    std::ofstream azi_file(file_name_trunk + ".azi_dir");
    std::ofstream normal_file(file_name_trunk + ".normal");

    for (int n = 0; n < 3*NBLOB; n++){
      blob_ref_file << blob_references[n] << " ";
      polar_file << polar_dir_refs[n] << " ";
      azi_file << azi_dir_refs[n] << " ";
      normal_file << normal_refs[n] << " ";

    }

    blob_ref_file.close();
    polar_file.close();
    azi_file.close();
    normal_file.close();

    std::cout << "...done!" << std::endl;

  };

  void seed_filaments(Real *const filament_references, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs){

    std::string file_name_trunk = GENERATRIX_FILE_NAME+std::to_string(NFIL);

    #if EQUATORIAL_SEEDING

      file_name_trunk += "_equatorial";

    #elif PLATY_SEEDING

      file_name_trunk += "_platy";

    #endif

    std::cout << std::endl << std::endl << "Generating seeding files " << "'" << file_name_trunk << ".seed'" << std::endl;
    
    shape_fourier_description shape;

    #if EQUATORIAL_SEEDING // TODO: Make this work for non-spheres.

      std::cout << "Seeding locations according to a spiral pattern..." << std::endl;

      

      const Real theta_max = 0.5/Real(R_OVER_L); // = 0.5 * 2*PI*L/(2*PI*R), meaning the width of the band is L as measured across the sphere surface.
      const Real h = 2.0*std::sin(theta_max);
      const int num_fils_for_sphere = std::ceil(NFIL*2.0/h);
      const int offset = std::round(0.5*(num_fils_for_sphere - NFIL));

      Real phi = 0.0;

      for (int n = 0; n < NFIL; n++){

        filament_references[3*n + 2] = 2.0*(n + offset)/Real(num_fils_for_sphere - 1) - 1.0;
        const Real r = std::sqrt(1.0 - filament_references[3*n + 2]*filament_references[3*n + 2]);
        filament_references[3*n] = r*std::cos(phi);
        filament_references[3*n + 1] = r*std::sin(phi);

        phi += 3.6/(r*std::sqrt(num_fils_for_sphere));

      }

    #elif PLATY_SEEDING // TODO: Make this work for non-spheres.

    

      std::cout << "Seeding..." << std::endl;

      const Real theta_max = 0.5/Real(R_OVER_L); // = 0.5 * 2*PI*L/(2*PI*R), meaning the width of the band is L as measured across the sphere surface.
      const Real h = 2.0*std::sin(theta_max);
      const int num_fils_in_main_band = std::round(0.9*NFIL); // Put 90% in the main band
      const int num_fils_for_sphere = std::ceil(num_fils_in_main_band*2.0/h);
      const int offset = std::round(0.5*(num_fils_for_sphere - num_fils_in_main_band));

      // Seed the main band
      Real phi = 0.0;

      for (int n = 0; n < num_fils_in_main_band; n++){

        filament_references[3*n + 2] = 2.0*(n + offset)/Real(num_fils_for_sphere - 1) - 1.0;
        const Real r = std::sqrt(1.0 - filament_references[3*n + 2]*filament_references[3*n + 2]);
        filament_references[3*n] = r*std::cos(phi);
        filament_references[3*n + 1] = r*std::sin(phi);

        phi += 3.6/(r*std::sqrt(num_fils_for_sphere));

      }

      // Seed the remaining 10% at the bottom of the swimmer
      for (int n = num_fils_in_main_band; n < NFIL; n++){

        filament_references[3*n + 2] = -0.9; // Some fixed location for the bottom band
        const Real r = std::sqrt(1.0 - filament_references[3*n + 2]*filament_references[3*n + 2]);
        phi = 2.0*PI*Real(n - num_fils_in_main_band)/Real(NFIL - num_fils_in_main_band);
        filament_references[3*n] = r*std::cos(phi);
        filament_references[3*n + 1] = r*std::sin(phi);

      }

    #elif UNIFORM_SEEDING

      std::cout << "Seeking an equal-area distribution for the filaments..." << std::endl;

      equal_area_seeding(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape);

    #elif HEXAGONAL_WALL_SEEDING
      std::cout << "Seeking an hexagonal placement for the fils..." << std::endl;
   
      // Real fil_grid_step_x;
      // Real blob_step;
      // int fil_grid_dim_x;
      // int hex_num;
      // Real rev_ratio;

      Real disc_radius;

      hexagonal_seeding(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape, 
                        FIL_SPACING, FIL_X_DIM, HEX_NUM, REV_RATIO);
      
      for (int fil_id=0; fil_id < NFIL; fil_id++){
        filament_references[3*fil_id + 2] = 2.1*BASE_HEIGHT_ABOVE_SURFACE;
      }

    #elif CENTRIC_WALL_SEEDING
      std::cout << "Seeking an centric placement for the fils..." << std::endl;
      Real disc_radius = FIL_SPACING*my_sqrt(NFIL);
      // Real disc_radius;
      //   std::ifstream in("separation.dat"); // using the 5th number as input
      //   for (int di=0; di<5; di++){
      //     in >> disc_radius;
      //   }

      equal_area_seeding_centric(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape, disc_radius, 0.04);
      
      for (int fil_id=0; fil_id < NFIL; fil_id++){
        filament_references[3*fil_id + 2] = 2.1*BASE_HEIGHT_ABOVE_SURFACE;
      }
    
    #elif ICOSA_SEEDING

      std::cout << "Seeking a icosahedron distribution for the filaments..." << std::endl;
      const std::string ico_file = SIMULATION_FILPLACEMENT_NAME;
      
      icosa_seeding(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape, ico_file);
    
    
    #elif MISMATCH_SEEDING

      std::cout << "Seeking a mismatched distribution for the filaments..." << std::endl;

      mismatch_seeding(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape);

    #elif UNIFORM_SEEDING_POLE

      std::cout << "Seeking an equal-area distribution (repulsion at poles) for the filaments..." << std::endl;

      if (PAIR){
        equal_area_seeding_poles_pair(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape);
      }else{
        equal_area_seeding_poles(filament_references, polar_dir_refs, azi_dir_refs, normal_refs, NFIL, shape);
      }
      
    #endif

    std::ofstream fil_ref_file(file_name_trunk + ".seed");
    std::ofstream polar_file(file_name_trunk + ".polar_dir");
    std::ofstream azi_file(file_name_trunk + ".azi_dir");
    std::ofstream normal_file(file_name_trunk + ".normal");

    for (int n = 0; n < 3*NFIL; n++){

      fil_ref_file << filament_references[n] << " ";
      polar_file << polar_dir_refs[n] << " ";
      azi_file << azi_dir_refs[n] << " ";
      normal_file << normal_refs[n] << " ";

    }

    fil_ref_file.close();
    polar_file.close();
    azi_file.close();
    normal_file.close();

    std::cout << "...done!" << std::endl; 

  };

  void check_seeding(Real *const filament_references, Real *const polar_dir_refs, Real *const azi_dir_refs, Real *const normal_refs){
    
    Real *X;
    shape_fourier_description shape;

    cudaMallocManaged(&X, 3*sizeof(Real));

    #if UNIFORM_SEEDING_POLE

      std::cout << "Checking seeding..." << std::endl;
    
      for (int n = 0; n < NFIL; n++){

        X[0] = filament_references[3*n];
        X[1] = filament_references[3*n+1];
        X[2] = filament_references[3*n+2];

        const Real theta = std::atan2(std::sqrt(X[0]*X[0] + X[1]*X[1]), X[2]);
        const Real phi = std::atan2(X[1], X[0]);

        matrix frame = shape.full_frame(theta, phi);

        // printf("fil %d (%.4f, %.4f) ref (%.4f %.4f %.4f) polar (%.4f %.4f %.4f) azi (%.4f %.4f %.4f) normal (%.4f %.4f %.4f)\n",
        // n, theta, phi, X[0], X[1], X[2],
        // frame(0), frame(1), frame(2),
        // frame(3), frame(4), frame(5),
        // frame(6), frame(7), frame(8));

        printf("fil %d (%.4f, %.4f) ref (%.4f %.4f %.4f) polar (%.4f %.4f %.4f) azi (%.4f %.4f %.4f) normal (%.4f %.4f %.4f)\n",
        n, theta, phi, X[0], X[1], X[2],
        polar_dir_refs[3*n], polar_dir_refs[3*n+1], polar_dir_refs[3*n+2],
        azi_dir_refs[3*n], azi_dir_refs[3*n+1], azi_dir_refs[3*n+2],
        normal_refs[3*n], normal_refs[3*n+1], normal_refs[3*n+2]);

      }

      

      

    #else

      std::cout << "Cannot check seeding of this type." << std::endl;

    #endif
    
    

  };

#endif
