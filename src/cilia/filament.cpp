// filament.cpp

#include <random>
#include <cmath>
#include <string>
#include "filament.hpp"
#include "../../config.hpp"

matrix body_frame_moment_lie_derivative(const quaternion& q1, const quaternion& q2, const bool upper){

  matrix A(4,3);

  if (upper){

    q2.psi_mat(A);
    A /= DL;

  } else {

    q1.psi_mat(A);
    A /= -DL;

  }

  quaternion midq = midpoint_quaternion(q1, q2);
  midq.conj_in_place();

  matrix mult_mat = midq.left_mult_mat();

  matrix dMdu = mult_mat*A;

  midq = (q2-q1)/DL; // dq/ds, but re-use the memory of midq
  midq.left_mult_mat(mult_mat); // re-use mult_mat memory

  A *= upper ? 0.5*DL : -0.5*DL;

  dMdu += mult_mat*A;

  matrix out = 2.0*dMdu.get_block(1,0,3,3); // throw away top row

  // We know the matrix is 3x3, so it's simple to just write out the multiplication
  // by diag(KT, KB, KB) explicitly, and we can do so to exploit the column-major
  // ordering of the data.
  out(0) *= KT;
  out(1) *= KB;
  out(2) *= KB;
  out(3) *= KT;
  out(4) *= KB;
  out(5) *= KB;
  out(6) *= KT;
  out(7) *= KB;
  out(8) *= KB;

  return out;

}

filament::~filament(){}

filament::filament(){}

void filament::initial_setup(const Real *const base_pos,
                              const Real *const dir_in,
                              const Real *const strain_twist_in,
                              const Real *const data_from_file,
                              Real *const x_address,
                              Real *const f_address,
                              const int fil_id,
                              quaternion rigidbody_q){

  id = fil_id;

  segments = std::vector<segment>(NSEG);

  f = f_address;

  // Tell the segments where to store their current positions
  for (int n = 0; n < NSEG; n++){

    segments[n].x = &x_address[3*n];

  }

  strain_twist[0] = strain_twist_in[0];
  strain_twist[1] = strain_twist_in[1];
  strain_twist[2] = strain_twist_in[2];

  #if (READ_INITIAL_CONDITIONS_FROM_BACKUP && !NO_CILIA_SQUIRMER)

    int p = 0;

    #if PRESCRIBED_CILIA

      phase = data_from_file[p++];
      phase_dot = data_from_file[p++];
      omega0 = data_from_file[p++];

      body_qm1.scalar_part = data_from_file[p++];
      body_qm1.vector_part[0] = data_from_file[p++];
      body_qm1.vector_part[1] = data_from_file[p++];
      body_qm1.vector_part[2] = data_from_file[p++];

      body_q.scalar_part = data_from_file[p++];
      body_q.vector_part[0] = data_from_file[p++];
      body_q.vector_part[1] = data_from_file[p++];
      body_q.vector_part[2] = data_from_file[p++];

      #if DYNAMIC_SHAPE_ROTATION // We cannot resume from backup if we're writing the generalised forces, so we can ignore the "|| WRITE_GENERALISED_FORCES" that appears elsewhere.

        shape_rotation_angle = data_from_file[p++];
        shape_rotation_angle_dot = data_from_file[p++];

        vel_dir_angle = std::vector<Real>(3*NSEG);

      #else

        // Increment p past the angle we don't need.
        // It shouldn't really be allowed to resume a fixed-angle sim from a dynamic-angle one, because if the angle is
        // fixed we assume it's zero rather than whatever value we're skipping here; this increment is just to stop the code from
        // blowing up if someone happens to try it. I could change it to allow the angle to be fixed at a general value,
        // but it's not a case I'm interested in running, and moreover it raises the possible need for angle-dependent
        // generalised forces for the phase variable.
        p++;
        p++;

      #endif

      vel_dir_phase = std::vector<Real>(3*NSEG);

      #if FIT_TO_DATA_BEAT

        fill_fourier_coeff_mats();

      #elif BUILD_A_BEAT

        beat_switch_theta = std::acos(0.5*SCALED_BEAT_AMPLITUDE);

      #endif

    #else

      for (int n = 0; n < NSEG; n++){

        segments[n].q.scalar_part = data_from_file[p++];
        segments[n].q.vector_part[0] = data_from_file[p++];
        segments[n].q.vector_part[1] = data_from_file[p++];
        segments[n].q.vector_part[2] = data_from_file[p++];

        segments[n].qm1 = segments[n].q;

        segments[n].x[0] = data_from_file[p++];
        segments[n].x[1] = data_from_file[p++];
        segments[n].x[2] = data_from_file[p++];
        segments[n].xm1[0] = data_from_file[p++];
        segments[n].xm1[1] = data_from_file[p++];
        segments[n].xm1[2] = data_from_file[p++];
        segments[n].xm2[0] = data_from_file[p++];
        segments[n].xm2[1] = data_from_file[p++];
        segments[n].xm2[2] = data_from_file[p++];

        segments[n].u[0] = data_from_file[p++];
        segments[n].u[1] = data_from_file[p++];
        segments[n].u[2] = data_from_file[p++];
        segments[n].um1[0] = data_from_file[p++];
        segments[n].um1[1] = data_from_file[p++];
        segments[n].um1[2] = data_from_file[p++];

      }

      tether_lambda[0] = data_from_file[p++];
      tether_lambda[1] = data_from_file[p++];
      tether_lambda[2] = data_from_file[p++];
      tether_lambdam1[0] = data_from_file[p++];
      tether_lambdam1[1] = data_from_file[p++];
      tether_lambdam1[2] = data_from_file[p++];
      tether_lambdam2[0] = data_from_file[p++];
      tether_lambdam2[1] = data_from_file[p++];
      tether_lambdam2[2] = data_from_file[p++];

      clamp_lambda[0] = data_from_file[p++];
      clamp_lambda[1] = data_from_file[p++];
      clamp_lambda[2] = data_from_file[p++];
      clamp_lambdam1[0] = data_from_file[p++];
      clamp_lambdam1[1] = data_from_file[p++];
      clamp_lambdam1[2] = data_from_file[p++];
      clamp_lambdam2[0] = data_from_file[p++];
      clamp_lambdam2[1] = data_from_file[p++];
      clamp_lambdam2[2] = data_from_file[p++];

      lambda = std::vector<Real>(3*(NSEG-1));
      lambdam1 = std::vector<Real>(3*(NSEG-1));
      lambdam2 = std::vector<Real>(3*(NSEG-1));

      for (int n = 0; n < NSEG-1; n++){

        lambda[3*n] = data_from_file[p++];
        lambda[3*n + 1] = data_from_file[p++];
        lambda[3*n + 2] = data_from_file[p++];

        lambdam1[3*n] = data_from_file[p++];
        lambdam1[3*n + 1] = data_from_file[p++];
        lambdam1[3*n + 2] = data_from_file[p++];

        lambdam2[3*n] = data_from_file[p++];
        lambdam2[3*n + 1] = data_from_file[p++];
        lambdam2[3*n + 2] = data_from_file[p++];

      }

      inverse_jacobian = matrix(6*NSEG, 6*NSEG);

      #if (CILIA_TYPE != 1)

        elastic_clamping_block1 = matrix(3,3);
        elastic_clamping_block2 = matrix(3,3);

      #endif

    #endif

  #else

    const Real dir_norm = sqrt(dir_in[0]*dir_in[0] + dir_in[1]*dir_in[1] + dir_in[2]*dir_in[2]);
    Real dir[3] = {dir_in[0]/dir_norm, dir_in[1]/dir_norm, dir_in[2]/dir_norm};

    quaternion qtemp(dir[0], 0.0, -dir[2], dir[1]);
    qtemp.sqrt_in_place_ortho(dir);

    #if PRESCRIBED_CILIA

      std::random_device rd_spread{};
      std::mt19937 gen_spread{rd_spread()};
      std::normal_distribution<Real> d(0,1);
      Real noise = OMEGA_SPREAD*d(gen_spread);

      omega0 = 2.0*PI + noise;

      // std::cout << id << "   " << omega0 << std::endl;

      if (PAIR==1){
        if (id/NPAIR == 1){
          omega0 *= PAIR_DP;
        }
      }
      
      #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

        shape_rotation_angle = 0.0;

      #endif

      #if WRITE_GENERALISED_FORCES

        phase = 0.0;

      #else

        #if (CILIA_IC_TYPE==0)

          phase = 0.0;

        #elif (CILIA_IC_TYPE==1)

          // Random phases
          std::random_device rd{};
          std::mt19937 gen{rd()};
          std::uniform_real_distribution<Real> distribution(0.0, 2.0*PI);
          phase = distribution(gen);

        #elif (CILIA_IC_TYPE==2)

          #if INFINITE_PLANE_WALL

            // The above quaternion maps the y-axis to itself, so we simply measure the MCW angle against the y-axis.
            phase = -2.0*PI*(base_pos[1]*std::cos(CILIA_MCW_ANGLE) - base_pos[0]*std::sin(CILIA_MCW_ANGLE))/CILIA_MCW_WAVELENGTH;

          #elif SURFACE_OF_REVOLUTION_BODIES

            const Real phi = atan2(base_pos[1], base_pos[0]);

            // TODO: Make this meaningful for non-spheres.
            phase = phi*PI*AXIS_DIR_BODY_LENGTH/CILIA_MCW_WAVELENGTH; // = phi * circumference / wavelength, if the body is a sphere.
            
            if (CILIA_MCW_ANGLE > 0){

              // Ignoring any tilt we introduce, the effective stroke will be 'downwards'. If we want positive angles to indicate the wave moving to the left (i.e. anti-clockwise) of this,
              // the MCW shoulds travel in the direction of increasing phi, meaning that the initial phases should decrease as phi increases.

              phase *= -1.0;

            }

          #endif

        #elif (CILIA_IC_TYPE==3)
          // WARNING this is only correct if a spherical body is initialised at the origin!!
          const Real phi = atan2(base_pos[1], base_pos[0]);
          const Real theta = acos(base_pos[2]/(sqrt(base_pos[0]*base_pos[0]+
                                                    base_pos[1]*base_pos[1]+
                                                    base_pos[2]*base_pos[2])));
                                                    
          Real k = 0.0;
          Real v = 0.0;
          std::ifstream in("input/ishikawa_wave/ishikawa.dat"); // input
          in >> k;
          in >> v;
          phase = Real(2.0)*PI*( sin(k*theta/2.0) + sin(v*phi/4.0) );
          in.close();

        #elif (CILIA_IC_TYPE==5)
          std::random_device rd{};
          std::mt19937 gen{rd()};
          std::uniform_real_distribution<Real> distribution(0.0, 2.0*PI);
          phase = distribution(gen);
          
          // Try to read from file
          std::ifstream input_file(SIMULATION_ICSTATE_NAME);
          if (input_file.is_open()) {
            if(fil_id == 0){
              std::cout << "Reading all states from file: " << SIMULATION_ICSTATE_NAME << std::endl;
            }
            float bin;
            input_file >> bin;
            input_file >> bin;
            Real buffer;
            for (int fpos = 0; fpos < 2*NFIL; fpos++){
              input_file >> buffer;
              if(fpos == fil_id){
                phase = buffer;
              }
              if(fpos == NFIL + fil_id){
                shape_rotation_angle = buffer;
              }
            }
            input_file.close();
          }else{
            if(fil_id == 0){
              std::cout << "No true_states file found: " << SIMULATION_ICSTATE_NAME << std::endl;
            }
          }
        #elif (CILIA_IC_TYPE==6)
          // WARNING this is only correct if a spherical body is initialised at the origin!!
          Real phi = atan2(base_pos[1], base_pos[0]);
          Real theta = acos(base_pos[2]/(sqrt(base_pos[0]*base_pos[0]+
                                                    base_pos[1]*base_pos[1]+
                                                    base_pos[2]*base_pos[2])));
          if (base_pos[0] == 0 && base_pos[1] == 0 && base_pos[2] == 0) {
            phi = 0;
            theta = 0;
          }


          // std::cout << base_pos[0] << "   " << base_pos[1] << "   " << base_pos[2] << "   " << std::endl;
          // std::cout << phi << "    " << theta << std::endl;

          phase = Real(2.0)*WAVNUM*theta + WAVNUM_DIA*phi;

        #endif

        #if SURFACE_OF_REVOLUTION_BODIES

          //const Real phi = atan2(base_pos[1], base_pos[0]);

          // Reduce the speed of the MCW on dorsal side of the swimmer
          //omega0 = (phi >= 0.0) ? 2.0*PI/1.7262 : 2.0*PI; // Uses T = 1.

        #endif
      
      #endif

      #if BICILIA
        phase2 = phase + PAIR_DP*2.0*PI;
      #elif BICILIA_LONGT
        phase2 = PAIR_DP*phase;
      #endif

      // qtemp maps x to the surface normal, which we want, but we also need to rotate about the surface normal to align
      // the 'normal-to-filament-centerline' vector at the base (i.e. the second frame vector) with the polar direction.
      Real base_normal[3];
      qtemp.normal(base_normal);
      // here base_normal is a vector that is orthogonal to the filament tangent.
      // n' in appendix A.1

      // Here we are getting the polar_dir using trivial vector analysis.
      // polar_dir is a vector, tangent to the spherical body towards the posterior
      // because (0,0, -1) is used in the vector analysis
      Real polar_dir[3] = {Real(0.0), Real(0.0), -Real(1.0)};
      const Real dir_dot_polar_dir = dir[0]*polar_dir[0] + dir[1]*polar_dir[1] + dir[2]*polar_dir[2];
      polar_dir[0] -= dir_dot_polar_dir*dir[0];
      polar_dir[1] -= dir_dot_polar_dir*dir[1];
      polar_dir[2] -= dir_dot_polar_dir*dir[2];
      
      const Real norm_polar_dir = sqrt(polar_dir[0]*polar_dir[0] + polar_dir[1]*polar_dir[1] + polar_dir[2]*polar_dir[2]);
      // here polar_dir becomes a vector that is orthogonal to dir.
      // therefore nothing 

      quaternion q2;
      quaternion q2_before;

      if (norm_polar_dir > 0.0){

        polar_dir[0] /= norm_polar_dir;
        polar_dir[1] /= norm_polar_dir;
        polar_dir[2] /= norm_polar_dir;

        q2.scalar_part = polar_dir[0]*base_normal[0] + polar_dir[1]*base_normal[1] + polar_dir[2]*base_normal[2];
        q2.vector_part[0] = base_normal[1]*polar_dir[2] - base_normal[2]*polar_dir[1];
        q2.vector_part[1] = base_normal[2]*polar_dir[0] - base_normal[0]*polar_dir[2];
        q2.vector_part[2] = base_normal[0]*polar_dir[1] - base_normal[1]*polar_dir[0];

        q2_before = q2;
        
        q2.sqrt_in_place_ortho(base_normal);

      }
      // If false, then we've managed to seed a filament at the pole.
      // Ideally I should have the seeding functions do something to ensure this can't happen.
      // If it has happened, then the best we can do is leave the beat direction as it is.
      // The default constructor will make q2 the identity, so this is what will happen here.

      // Introduce tilt
      const Real sn = sin(0.5*TILT_ANGLE);
      const Real cs = cos(0.5*TILT_ANGLE);
      q2 = quaternion(cs, sn*dir[0], sn*dir[1], sn*dir[2])*q2;
      
      body_qm1 = q2*qtemp;
      body_q_ref = body_qm1;
      // rotate to get the correct body_q if the rigidbody_q is not default {1, 0, 0, 0}
      body_qm1 = rigidbody_q*body_qm1;
      body_q = body_qm1;

      // DEBUGGING BEGINS
      // Real body_q_dir[3];
      // body_q.normal(body_q_dir);
      
      // if(abs(polar_dir[0]-body_q_dir[0]) > 0.00001 || abs(polar_dir[1]-body_q_dir[1]) > 0.00001 || abs(polar_dir[2]-body_q_dir[2]) > 0.00001){
      //   printf("fil %d q2_before_sqrt (%.16f %.16f %.16f %.16f) \n",
      //   fil_id, q2(0), q2(1), q2(2), q2(3));
      //   printf("fil %d polar (%.4f %.4f %.4f) body_q_dir (%.4f %.4f %.4f) q2 (%.4f %.4f %.4f %.4f) qtemp (%.4f %.4f %.4f %.4f) body_q (%.16f %.16f %.16f %.16f)  \n",
      //   fil_id, polar_dir[0], polar_dir[1], polar_dir[2],
      //   body_q_dir[0], body_q_dir[1], body_q_dir[2],
      //   q2(0), q2(1), q2(2), q2(3),
      //   qtemp(0), qtemp(1), qtemp(2), qtemp(3),
      //   body_q(0), body_q(1), body_q(2), body_q(3) );
      // }


      // DEBUGGING ENDE

      #if FIT_TO_DATA_BEAT

        fill_fourier_coeff_mats();

      #elif BUILD_A_BEAT

        beat_switch_theta = std::acos(0.5*SCALED_BEAT_AMPLITUDE);

      #endif

      vel_dir_phase = std::vector<Real>(3*NSEG);

      #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

        vel_dir_angle = std::vector<Real>(3*NSEG);

        shape_rotation_angle_dot = 0.0; // Initial guess. Shouldn't actually be necessary if we entered here because of WRITE_GENERALISED_FORCES?

      #endif

      // This is the definition if the phase evolution is prescribed, and the initial guess otherwise.
      // We do this assignment at the end in case we've modified omega0 from 2pi at some point.
      phase_dot = omega0;

    #else

      inverse_jacobian = matrix(6*NSEG, 6*NSEG);

      Real seg_pos[3] = {base_pos[0], base_pos[1], base_pos[2]};
      Real seg_u[3] = {0.0, 0.0, 0.0};

      for (int i = 0; i < NSEG; i++) {

        segments[i].initial_setup(seg_pos, qtemp, seg_u);

        seg_pos[0] += DL*dir[0];
        seg_pos[1] += DL*dir[1];
        seg_pos[2] += DL*dir[2];

      }

      #if CONSTANT_BASE_ROTATION

        qtemp.tangent(segments[0].u);
        segments[0].u *= BASE_ROTATION_RATE*DT;
        segments[0].um1 = segments[0].u;
        segments[0].um2 = segments[0].u;

      #endif

      lambda = std::vector<Real>(3*(NSEG-1));
      lambdam1 = std::vector<Real>(3*(NSEG-1));
      lambdam2 = std::vector<Real>(3*(NSEG-1));

      for (int i = 0; i < 3*(NSEG-1); i++){

        lambda[i] = 0.0;
        lambdam1[i] = 0.0;
        lambdam2[i] = 0.0;

      }

      tether_lambda[0] = 0.0;
      tether_lambda[1] = 0.0;
      tether_lambda[2] = 0.0;

      #if INSTABILITY_CILIA

        clamp_lambda[0] = 0.0;
        clamp_lambda[1] = 0.0;
        clamp_lambda[2] = 0.0;

        const Real pMag = 0.01*END_FORCE_MAGNITUDE;

        #if (CILIA_IC_TYPE==0)

          perturbation1[0] = pMag;
          perturbation1[1] = 0.0;
          perturbation1[2] = 0.0;

          perturbation2[0] = 0.0;
          perturbation2[1] = 0.0;
          perturbation2[2] = 0.0;

        #elif (CILIA_IC_TYPE==1)

          perturbation1[0] = pMag;
          perturbation1[1] = 0.0;
          perturbation1[2] = 0.0;

          perturbation2[0] = 0.0;
          perturbation2[1] = pMag;
          perturbation2[2] = 0.0;

        #elif (CILIA_IC_TYPE==2)

          std::random_device rd{};
          std::mt19937 gen{rd()};
          std::normal_distribution<Real> d(0,1);

          Real dir1[3] = {d(gen), d(gen), d(gen)};
          const Real norm_dir1 = sqrt(dir1[0]*dir1[0] + dir1[1]*dir1[1] + dir1[2]*dir1[2]);

          perturbation1[0] = pMag*dir1[0]/norm_dir1;
          perturbation1[1] = pMag*dir1[1]/norm_dir1;
          perturbation1[2] = pMag*dir1[2]/norm_dir1;

          perturbation2[0] = 0.0;
          perturbation2[1] = 0.0;
          perturbation2[2] = 0.0;

        #elif (CILIA_IC_TYPE==3)

          std::random_device rd{};
          std::mt19937 gen{rd()};
          std::normal_distribution<Real> d(0,1);

          Real dir1[3] = {d(gen), d(gen), d(gen)};
          const Real norm_dir1 = sqrt(dir1[0]*dir1[0] + dir1[1]*dir1[1] + dir1[2]*dir1[2]);
          Real dir2[3] = {d(gen), d(gen), d(gen)};
          const Real norm_dir2 = sqrt(dir2[0]*dir2[0] + dir2[1]*dir2[1] + dir2[2]*dir2[2]);

          perturbation1[0] = pMag*dir1[0]/norm_dir1;
          perturbation1[1] = pMag*dir1[1]/norm_dir1;
          perturbation1[2] = pMag*dir1[2]/norm_dir1;

          perturbation2[0] = pMag*dir2[0]/norm_dir2;
          perturbation2[1] = pMag*dir2[1]/norm_dir2;
          perturbation2[2] = pMag*dir2[2]/norm_dir2;

        #endif

      #elif GEOMETRIC_CILIA

        base_torque_magnitude_factor = -1.0/DRIVING_TORQUE_MAGNITUDE_RATIO;

        step_at_start_of_transition = TOTAL_TIME_STEPS+1;

        just_switched_from_fast_to_slow = false;
        just_switched_from_slow_to_fast = false;

        body_qm1 = quaternion(1.0, 0.0, 0.0, 0.0);
        body_q = body_qm1;

        my_vertical = dir;

        #if SADDLE_BODIES

          fast_beating_direction = {1.0, 0.0, 0.0};

        #elif SURFACE_OF_REVOLUTION_BODIES

          //TODO: Needs updating to reflect fact that body isn't necessarily a sphere anymore.

          fast_beating_direction = {0.0, 0.0, 1.0};
          fast_beating_direction -= dot(fast_beating_direction, dir)*dir; // Result can't be 0 as filaments aren't seeded on poles.
          fast_beating_direction = normalise(fast_beating_direction);

        #endif

      #elif CONSTANT_BASE_ROTATION

        clamp_lambda.zeros();

      #endif

      #if (CILIA_TYPE != 1)

        elastic_clamping_block1 = matrix(3,3);
        elastic_clamping_block2 = matrix(3,3);

      #endif

    #endif

  #endif

}

void filament::robot_arm(){

  Real t1[3], t2[3];

  segments[0].tangent(t1);

  for (int i = 1; i < NSEG; i++) {

    segments[i].tangent(t2);

    segments[i].x[0] = segments[i-1].x[0] + 0.5*DL*(t1[0] + t2[0]);
    segments[i].x[1] = segments[i-1].x[1] + 0.5*DL*(t1[1] + t2[1]);
    segments[i].x[2] = segments[i-1].x[2] + 0.5*DL*(t1[2] + t2[2]);

    

    t1[0] = t2[0];
    t1[1] = t2[1];
    t1[2] = t2[2];

  }

}

void filament::accept_state_from_rigid_body(const Real *const x_in, const Real *const u_in){

  segments[0].x[0] = x_in[0];
  segments[0].x[1] = x_in[1];
  segments[0].x[2] = x_in[2];

  #if (GEOMETRIC_CILIA || PRESCRIBED_CILIA)

    lie_exp(body_q, u_in);
    body_q *= body_qm1;

  #else

    // Write it this way to trigger the quaternion update too.
    const Real u_update[3] = {u_in[0] - segments[0].u[0], u_in[1] - segments[0].u[1], u_in[2] - segments[0].u[2]};
    segments[0].update(u_update);

  #endif

  #if !PRESCRIBED_CILIA

    robot_arm();

  #endif

}

#if PRESCRIBED_CILIA

  #if FIT_TO_DATA_BEAT

    void filament::fill_fourier_coeff_mats(){

      #if FULFORD_AND_BLAKE_BEAT || FULFORD_AND_BLAKE_BEAT_NO_WALL

        // See Table 1 of Ito, Omori and Ishikawa (2019).
        // We use a transposed layout because of how these matrices are applied.
        // Furthermore, we divide the values given by Ito et al. through by ~0.97475
        // so that the average curve length over a period is 1. The zero-mode cosine
        // coefficients are additionally multiplied by 0.5, allowing our coefficients
        // to be used in simple matrix multiplication. Lastly, our simulations are based
        // on the specific case g1 = e_y and g2 = e_x, so e.g. our Ax corresponds to their A2
        // and our Ay to their A1.

        Ay = matrix(4,3);
        Ay(0,0) = -3.3547e-01;
        Ay(1,0) = 4.0318e-01;
        Ay(2,0) = -9.9513e-02;
        Ay(3,0) = 8.1046e-02;
        Ay(0,1) = 4.0369e-01;
        Ay(1,1) = -1.5553e+00;
        Ay(2,1) = 3.2829e-02;
        Ay(3,1) = -3.0982e-01;
        Ay(0,2) = 1.0362e-01;
        Ay(1,2) = 7.3455e-01;
        Ay(2,2) = -1.2106e-01;
        Ay(3,2) = 1.4568e-01;

        Ax = matrix(4,3);
        Ax(0,0) = 9.7204e-01;
        Ax(1,0) = -1.8466e-02;
        Ax(2,0) = 1.6209e-01;
        Ax(3,0) = 1.0259e-02;
        Ax(0,1) = -2.8315e-01;
        Ax(1,1) = -1.2926e-01;
        Ax(2,1) = -3.4983e-01;
        Ax(3,1) = 3.5907e-02;
        Ax(0,2) = 4.9243e-02;
        Ax(1,2) = 2.6981e-01;
        Ax(2,2) = 1.9082e-01;
        Ax(3,2) = -6.8736e-02;

        By = matrix(3,3);
        By(0,0) = 2.9136e-01;
        By(1,0) = 6.1554e-03;
        By(2,0) = -6.0528e-02;
        By(0,1) = 1.0721e+00;
        By(1,1) = 3.2521e-01;
        By(2,1) = 2.3185e-01;
        By(0,2) = -1.0433e+00;
        By(1,2) = -2.8315e-01;
        By(2,2) = -2.0108e-01;
        

        Bx = matrix(3,3);
        Bx(0,0) = 1.9697e-01;
        Bx(1,0) = -5.1295e-02;
        Bx(2,0) = 1.2311e-02;
        Bx(0,1) = -5.1193e-01;
        Bx(1,1) = 4.3396e-01;
        Bx(2,1) = 1.4157e-01;
        Bx(0,2) = 3.4778e-01;
        Bx(1,2) = -3.3547e-01;
        Bx(2,2) = -1.1695e-01;

      #elif VOLVOX_BEAT

        // Same as Fulford-Blake beat but with newly fitted coefficients

        Ay = matrix(4,4);
        Ay(0,0) = -0.01665535;
        Ay(1,0) = 0.17505739;
        Ay(2,0) = -0.12408600;
        Ay(3,0) = -0.16672661;
        Ay(0,1) = -0.27457190;
        Ay(1,1) = -2.09412076;
        Ay(2,1) = 1.62438155;
        Ay(3,1) = 0.90896038;
        Ay(0,2) = 1.08406999;
        Ay(1,2) = 1.99556763;
        Ay(2,2) = -3.21015214;
        Ay(3,2) = -1.50867276;
        Ay(0,3) = -0.56102820;
        Ay(1,3) = -0.50179439;
        Ay(2,3) = 1.70178281;
        Ay(3,3) = 0.74387795;
        Ax = matrix(4,4);
        Ax(0,0) = 1.00490080;
        Ax(1,0) = 0.00314934;
        Ax(2,0) = 0.18942274;
        Ax(3,0) = -0.11162123;
        Ax(0,1) = -0.19958656;
        Ax(1,1) = -0.22239636;
        Ax(2,1) = -1.24525818;
        Ax(3,1) = 1.07262966;
        Ax(0,2) = -0.20422343;
        Ax(1,2) = 1.18659976;
        Ax(2,2) = 1.94063304;
        Ax(3,2) = -2.15142914;
        Ax(0,3) = 0.10887946;
        Ax(1,3) = -0.87409016;
        Ax(2,3) = -0.78632572;
        Ax(3,3) = 1.20544741;
        By = matrix(3,4);
        By(0,0) = -0.06343445;
        By(1,0) = -0.28828336;
        By(2,0) = 0.02187088;
        By(0,1) = 2.15509751;
        By(1,1) = 1.45223183;
        By(2,1) = -0.33070891;
        By(0,2) = -4.19609063;
        By(1,2) = -1.56580487;
        By(2,2) = 0.94906955;
        By(0,3) = 2.05489100;
        By(1,3) = 0.44541361;
        By(2,3) = -0.66157093;
        Bx = matrix(3,4);
        Bx(0,0) = 0.09140608;
        Bx(1,0) = -0.09670739;
        Bx(2,0) = -0.16451150;
        Bx(0,1) = -0.54804089;
        Bx(1,1) = 1.14634486;
        Bx(2,1) = 0.98846449;
        Bx(0,2) = 1.40643137;
        Bx(1,2) = -2.60675971;
        Bx(2,2) = -1.49147276;
        Bx(0,3) = -0.60109684;
        Bx(1,3) = 1.55525152;
        Bx(2,3) = 0.64896090;

      #elif FULFORD_AND_BLAKE_BEAT_ORIGINAL

        // Same as Fulford-Blake beat but with unaltered coefficients

        Ay = matrix(4,3);
        Ay(0,0) = -0.654;
        Ay(1,0) = 0.393;
        Ay(2,0) = -0.097;
        Ay(3,0) = 0.079;
        Ay(0,1) = 0.787;
        Ay(1,1) = -1.516;
        Ay(2,1) = 0.032;
        Ay(3,1) = -0.302;
        Ay(0,2) = 0.202;
        Ay(1,2) = 0.716;
        Ay(2,2) = -0.118;
        Ay(3,2) = 0.142;

        Ax = matrix(4,3);
        Ax(0,0) = 1.895;
        Ax(1,0) = -0.018;
        Ax(2,0) = 0.158;
        Ax(3,0) = 0.010;
        Ax(0,1) = -0.552;
        Ax(1,1) = -0.126;
        Ax(2,1) = -0.341;
        Ax(3,1) = 0.035;
        Ax(0,2) = 0.096;
        Ax(1,2) = 0.263;
        Ax(2,2) = 0.186;
        Ax(3,2) = -0.067;
        

        By = matrix(3,3);
        By(0,0) = 0.284;
        By(1,0) = 0.006;
        By(2,0) = -0.059;
        By(0,1) = 1.045;
        By(1,1) = 0.317;
        By(2,1) = 0.226;
        By(0,2) = -1.017;
        By(1,2) = -0.276;
        By(2,2) = -0.196;
        

        Bx = matrix(3,3);
        Bx(0,0) = 0.192;
        Bx(1,0) = -0.050;
        Bx(2,0) = 0.012;
        Bx(0,1) = -0.499;
        Bx(1,1) = 0.423;
        Bx(2,1) = 0.138;
        Bx(0,2) = 0.339;
        Bx(1,2) = -0.327;
        Bx(2,2) = -0.114;

      #elif CORAL_LARVAE_BEAT

        // These coefficients are scaled so that the average cilium length over a period is 1.

        Ay = matrix(4,3);
        Ay(0,0) = -2.766076e-01; Ay(0,1) = 6.368662e-01; Ay(0,2) = -8.513947e-02;
        Ay(1,0) = 3.091205e-01; Ay(1,1) = -1.993712e+00; Ay(1,2) = 1.091847e+00;
        Ay(2,0) = -2.952120e-01; Ay(2,1) = 7.189021e-01; Ay(2,2) = -6.328589e-01;
        Ay(3,0) = -9.868018e-02; Ay(3,1) = 3.312812e-01; Ay(3,2) = -2.578206e-01;

        Ax = matrix(4,3);
        Ax(0,0) = 9.034276e-01; Ax(0,1) = -6.634681e-01; Ax(0,2) = 2.016123e-01;
        Ax(1,0) = -2.468656e-01; Ax(1,1) = 1.215869e+00; Ax(1,2) = -5.867236e-01;
        Ax(2,0) = 3.075836e-01; Ax(2,1) = -6.966322e-01; Ax(2,2) = 4.554729e-01;
        Ax(3,0) = 2.378489e-02; Ax(3,1) = -1.355582e-01; Ax(3,2) = 7.315831e-02;

        By = matrix(3,3);
        By(0,0) = 9.165546e-01; By(0,1) = 8.848315e-02; By(0,2) = -4.838832e-01;
        By(1,0) = -2.946211e-01; By(1,1) = 1.057964e+00; By(1,2) = -6.335721e-01;
        By(2,0) = 6.122744e-02; By(2,1) = -2.319094e-01; By(2,2) = 1.906292e-01;

        Bx = matrix(3,3);
        Bx(0,0) = 1.336807e-01; Bx(0,1) = -6.807852e-01; Bx(0,2) = 6.008592e-01;
        Bx(1,0) = 8.146824e-02; Bx(1,1) = 3.472676e-01; Bx(1,2) = -2.744220e-01;
        Bx(2,0) = 3.615272e-02; Bx(2,1) = 8.619119e-02; Bx(2,2) = -6.122992e-02;

      #elif BICILIA or BICILIA_LONGT

        // Same as Fulford-Blake beat but with newly fitted coefficients

        Ay = matrix(4,4);
        Ay(0,0) = -0.01665535;
        Ay(1,0) = 0.17505739;
        Ay(2,0) = -0.12408600;
        Ay(3,0) = -0.16672661;
        Ay(0,1) = -0.27457190;
        Ay(1,1) = -2.09412076;
        Ay(2,1) = 1.62438155;
        Ay(3,1) = 0.90896038;
        Ay(0,2) = 1.08406999;
        Ay(1,2) = 1.99556763;
        Ay(2,2) = -3.21015214;
        Ay(3,2) = -1.50867276;
        Ay(0,3) = -0.56102820;
        Ay(1,3) = -0.50179439;
        Ay(2,3) = 1.70178281;
        Ay(3,3) = 0.74387795;
        Ax = matrix(4,4);
        Ax(0,0) = 1.00490080;
        Ax(1,0) = 0.00314934;
        Ax(2,0) = 0.18942274;
        Ax(3,0) = -0.11162123;
        Ax(0,1) = -0.19958656;
        Ax(1,1) = -0.22239636;
        Ax(2,1) = -1.24525818;
        Ax(3,1) = 1.07262966;
        Ax(0,2) = -0.20422343;
        Ax(1,2) = 1.18659976;
        Ax(2,2) = 1.94063304;
        Ax(3,2) = -2.15142914;
        Ax(0,3) = 0.10887946;
        Ax(1,3) = -0.87409016;
        Ax(2,3) = -0.78632572;
        Ax(3,3) = 1.20544741;
        By = matrix(3,4);
        By(0,0) = -0.06343445;
        By(1,0) = -0.28828336;
        By(2,0) = 0.02187088;
        By(0,1) = 2.15509751;
        By(1,1) = 1.45223183;
        By(2,1) = -0.33070891;
        By(0,2) = -4.19609063;
        By(1,2) = -1.56580487;
        By(2,2) = 0.94906955;
        By(0,3) = 2.05489100;
        By(1,3) = 0.44541361;
        By(2,3) = -0.66157093;
        Bx = matrix(3,4);
        Bx(0,0) = 0.09140608;
        Bx(1,0) = -0.09670739;
        Bx(2,0) = -0.16451150;
        Bx(0,1) = -0.54804089;
        Bx(1,1) = 1.14634486;
        Bx(2,1) = 0.98846449;
        Bx(0,2) = 1.40643137;
        Bx(1,2) = -2.60675971;
        Bx(2,2) = -1.49147276;
        Bx(0,3) = -0.60109684;
        Bx(1,3) = 1.55525152;
        Bx(2,3) = 0.64896090;

        #endif

    }

    void filament::fitted_shape_tangent(Real& tx, Real& ty, const Real s, const Real psi) const {

      const int num_degrees = Ax.num_cols;
      const int num_fourier_modes = Ax.num_rows;

      matrix svec(num_degrees,1);
      for(int i = 0; i < num_degrees; i++){
        svec(i) = float(i+1)*pow(s, i);
      }

      matrix cos_vec(1, num_fourier_modes);
      matrix sin_vec(1, num_fourier_modes-1);
      cos_vec(0) = 1.0;
      #if FULFORD_AND_BLAKE_BEAT_ORIGINAL
        cos_vec(0) = 0.5;
      #endif
      for (int n = 1; n < num_fourier_modes; n++){
        cos_vec(n) = std::cos(n*psi);
        sin_vec(n-1) = std::sin(n*psi);
      }

      tx = (cos_vec*Ax + sin_vec*Bx)*svec;
      ty = (cos_vec*Ay + sin_vec*By)*svec;

    }

    matrix filament::fitted_shape(const Real s, const Real psi, const Real z_displacement) const {

      matrix pos(3,1);
      pos(2) = z_displacement;

      const int num_degrees = Ax.num_cols;
      const int num_fourier_modes = Ax.num_rows;

      matrix svec(num_degrees,1);
      for(int i = 0; i < num_degrees; i++){
        svec(i) = pow(s, i+1);
      }

      matrix cos_vec(1, num_fourier_modes);
      matrix sin_vec(1, num_fourier_modes-1);
      cos_vec(0) = 1.0;
      #if FULFORD_AND_BLAKE_BEAT_ORIGINAL
        cos_vec(0) = 0.5;
      #endif
      for (int n = 1; n < num_fourier_modes; n++){
        cos_vec(n) = std::cos(n*psi);
        sin_vec(n-1) = std::sin(n*psi);
      }

      pos(0) = (cos_vec*Ax + sin_vec*Bx)*svec;
      pos(1) = (cos_vec*Ay + sin_vec*By)*svec;

      return pos;

    }

    matrix filament::fitted_shape_velocity_direction(const Real s, const Real psi) const {

      matrix dir(3,1);
      dir(2) = 0.0;

      const int num_degrees = Ax.num_cols;
      const int num_fourier_modes = Ax.num_rows;

      matrix svec(num_degrees,1);
      for(int i = 0; i < num_degrees; i++){
        svec(i) = pow(s, i+1);
      }

      matrix cos_vec(1, num_fourier_modes-1);
      matrix sin_vec(1, num_fourier_modes);
      sin_vec(0) = 0.0;
      for (int n = 1; n < num_fourier_modes; n++){
        cos_vec(n-1) = n*std::cos(n*psi);
        sin_vec(n) = -n*std::sin(n*psi);
      }

      dir(0) = (sin_vec*Ax + cos_vec*Bx)*svec;
      dir(1) = (sin_vec*Ay + cos_vec*By)*svec;

      return dir;

    }

    Real filament::fitted_curve_length(const Real s, const Real psi) const {

      Real length = 0.0;

      if (s > 0){

        Real tx, ty; // Components of shape tangent

        const int num_traps = 10*NSEG_PER_CILIA;
        const Real dl = 1.0/Real(num_traps);

        // Start with the whole trapeziums
        fitted_shape_tangent(tx, ty, 0, psi);
        Real f0 = sqrt(tx*tx + ty*ty);
        const Real floor_val = myfil_floor(s/dl);

        for (int n = 1; n <= floor_val; n++){

          fitted_shape_tangent(tx, ty, n*dl, psi);
          Real f1 = sqrt(tx*tx + ty*ty);

          length += 0.5*dl*(f0 + f1);

          f0 = f1;

        }

        // Interpolate between the mesh points to get the final value.
        // I want to do it like this, rather than have the mesh depend on the input s,
        // to ensure that the output is monotonic.
        const Real ceil_val = myfil_ceil(s/dl);

        if (ceil_val > floor_val){

          fitted_shape_tangent(tx, ty, myfil_ceil(s/dl)*dl, psi);
          Real f1 = sqrt(tx*tx + ty*ty);

          length += ((s/dl - floor_val)/(ceil_val - floor_val))*0.5*dl*(f0 + f1);

        }

      }

      return length;

    }

    void filament::find_fitted_shape_s(){
      
      s_to_use = std::vector<Real>(NSEG_PER_CILIA);
      s_to_use[0] = 0.0;
      s_to_use[NSEG_PER_CILIA-1] = 1.0;

      #if BICILIA

        s_to_use2 = std::vector<Real>(NSEG_PER_CILIA);
        s_to_use2[0] = 0.0;
        s_to_use2[NSEG_PER_CILIA-1] = 1.0;

      #elif BICILIA_LONGT

        s_to_use2 = std::vector<Real>(NSEG_PER_CILIA);
        s_to_use2[0] = 0.0;
        s_to_use2[NSEG_PER_CILIA-1] = 1.0;

      #endif

      bool bicilia = (BICILIA || BICILIA_LONGT);

      for (int fp = 0; fp < (bicilia ? 2 : 1); fp++){

        Real psi = (fp == 0 ? phase : phase2);

        #if WRITE_GENERALISED_FORCES

          std::ofstream s_values_file(reference_s_values_file_name(), std::ios::app);

          if (fp==0){

            s_values_file << s_to_use[0] << " ";

          }

          const Real total_length = fitted_curve_length(1.0, psi);

          for (int n = 1; n < NSEG_PER_CILIA-1; n++){

            const Real target_fraction = Real(n)/Real(NSEG_PER_CILIA-1);

            Real s_lower_bound = (fp==0 ? s_to_use[n-1] : s_to_use2[n-1]);
            Real s_upper_bound = 1.0;

            // Bisection method
            Real curr_s_estimate = 0.5*(s_lower_bound + s_upper_bound);
            Real curr_frac_estimate = fitted_curve_length(curr_s_estimate, psi)/total_length;

            while (std::abs(curr_frac_estimate - target_fraction) > 0.1/Real(NSEG_PER_CILIA)){

              if (curr_frac_estimate > target_fraction){

                s_upper_bound = curr_s_estimate;

              } else {

                s_lower_bound = curr_s_estimate;

              }

              curr_s_estimate = 0.5*(s_lower_bound + s_upper_bound);
              curr_frac_estimate = fitted_curve_length(curr_s_estimate, psi)/total_length;

            }
            // force to use uniform s
            // curr_s_estimate = target_fraction;

            if (fp==0){

              s_to_use[n] = curr_s_estimate;

              s_values_file << curr_s_estimate << " ";

            }
            else if (fp==1) {

              s_to_use2[n] = curr_s_estimate;

            }

          }

          if (fp==0){

            s_values_file << s_to_use[NSEG_PER_CILIA-1] << " ";
            s_values_file.close();

          }

        #else

          // Bilinear interpolation
          Real phi_index = 0.5*psi/PI; // = phase/(2*pi)
          phi_index -= myfil_floor(phi_index); // Map this ratio into [0,1)
          phi_index *= (*s_to_use_ref_ptr).num_rows;

          int phi_index_int_lower_bound = int(phi_index); // Rounds towards 0.

          matrix svec;

          if (phi_index_int_lower_bound == (*s_to_use_ref_ptr).num_rows){

            // This can only be true if phi_index == s_to_use_ref.num_rows
            // I don't think we can ever actually enter here unless rounding error messes things up,
            // but better safe than sorry.
            svec = (*s_to_use_ref_ptr).get_row(0);

          } else {

            // Otherwise, we're safely in the interior
            const matrix lower = (*s_to_use_ref_ptr).get_row(phi_index_int_lower_bound);
            const matrix upper = (phi_index_int_lower_bound == (*s_to_use_ref_ptr).num_rows - 1) ? (*s_to_use_ref_ptr).get_row(0) : (*s_to_use_ref_ptr).get_row(phi_index_int_lower_bound+1);
            svec = lower + (upper - lower)*(phi_index - phi_index_int_lower_bound);

          }

          for (int n = 1; n < NSEG_PER_CILIA-1; n++){

            Real s_index = ((*s_to_use_ref_ptr).num_cols - 1)*n/Real(NSEG_PER_CILIA-1);
            int s_index_lower_bound = int(s_index);

            if (fp==0){

              s_to_use[n] = svec(s_index_lower_bound) + (s_index - s_index_lower_bound)*(svec(s_index_lower_bound+1) - svec(s_index_lower_bound));
            
            }
            else if (fp==1) {

              s_to_use2[n] = svec(s_index_lower_bound) + (s_index - s_index_lower_bound)*(svec(s_index_lower_bound+1) - svec(s_index_lower_bound));
            
            }

          }

        #endif

      }

    }


  #elif BUILD_A_BEAT

    Real filament::build_a_beat_tangent_angle(const Real s) const {

      const Real wR = RECOVERY_STROKE_WINDOW_LENGTH*2.0*PI;
      const Real wE = EFFECTIVE_STROKE_LENGTH*2.0*PI;
      const Real phi_transition = wE + s*(2.0*PI - wE - wR); // Assumes 0 <= s <= 1

      //const Real shifted_phase = phase - 2.0*PI*std::floor(0.5*phase/PI);
      Real shifted_phase = phase - s*2.0*PI*ZERO_VELOCITY_AVOIDANCE_LENGTH;
      shifted_phase -= 2.0*PI*std::floor(0.5*shifted_phase/PI);

      Real delta;

      if (shifted_phase < wE){

        const Real temp = PI*(wE - shifted_phase)/wE;

        delta = temp - std::sin(temp)*std::cos(temp);

      } else if (shifted_phase < phi_transition){

        delta = 0.0;

      } else if (shifted_phase < phi_transition + wR){

        const Real temp = PI*(shifted_phase - phi_transition)/wR;

        delta = temp - std::sin(temp)*std::cos(temp);

      } else {

        delta = PI;

      }

      return beat_switch_theta + (PI - 2.0*beat_switch_theta)*delta/PI;

    }

    void filament::build_a_beat_tangent(matrix& t, const Real s) const {

      // Because of how we define quaternions to map the x-axis to the tangent, tx should be
      // the normal/vertical component of the reference tangent.
      t(0) = std::sin(build_a_beat_tangent_angle(s));
      t(1) = std::cos(build_a_beat_tangent_angle(s));
      t(2) = 0.0;

    }

    void filament::build_a_beat_tangent_phase_deriv(matrix& k, const Real s) const {

      const Real fac = 2.0*(PI - 2.0*beat_switch_theta);

      const Real wR = RECOVERY_STROKE_WINDOW_LENGTH*2.0*PI;
      const Real wE = EFFECTIVE_STROKE_LENGTH*2.0*PI;
      const Real phi_transition = wE + s*(2.0*PI - wE - wR); // Assumes 0 <= s <= 1

      //const Real shifted_phase = phase - 2.0*PI*std::floor(0.5*phase/PI);
      Real shifted_phase = phase - s*2.0*PI*ZERO_VELOCITY_AVOIDANCE_LENGTH;
      shifted_phase -= 2.0*PI*std::floor(0.5*shifted_phase/PI);

      Real delta_deriv;

      if (shifted_phase < wE){

        const Real temp = PI*(wE - shifted_phase)/wE;

        delta_deriv = std::sin(temp);
        delta_deriv *= -delta_deriv/wE;

      } else if (shifted_phase < phi_transition){

        delta_deriv = 0.0;

      } else if (shifted_phase < phi_transition + wR){

        const Real temp = PI*(shifted_phase - phi_transition)/wR;

        delta_deriv = std::sin(temp);
        delta_deriv *= delta_deriv/wR;

      } else {

        delta_deriv = 0.0;

      }

      k(0) = fac*delta_deriv*std::cos(build_a_beat_tangent_angle(s));
      k(1) = -fac*delta_deriv*std::sin(build_a_beat_tangent_angle(s));
      k(2) = 0.0;

    }

  #endif

#endif

void filament::initial_guess(const int nt, const Real *const x_in, const Real *const u_in){

  #if PRESCRIBED_CILIA

    if (nt > 0){

      phase += phase_dot*DT;

      #if BICILIA
        phase2 = phase + PAIR_DP*2.0*PI;
      #elif BICILIA_LONGT
        phase2 = PAIR_DP*phase;
      #endif

      #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

        shape_rotation_angle += shape_rotation_angle_dot*DT;

      #endif

    }

  #endif

  #if (GEOMETRIC_CILIA || PRESCRIBED_CILIA)

    body_qm1 = body_q;

  #endif

  #if !PRESCRIBED_CILIA

    for (int i = 0; i < NSEG; i++) {

      segments[i].initial_guess(nt);

    }

  #endif

  accept_state_from_rigid_body(x_in, u_in);

  #if PRESCRIBED_CILIA

    const matrix R = body_q.rot_mat();
    
    vel_dir_phase[0] = 0.0;
    vel_dir_phase[1] = 0.0;
    vel_dir_phase[2] = 0.0;

    #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

      vel_dir_angle[0] = 0.0;
      vel_dir_angle[1] = 0.0;
      vel_dir_angle[2] = 0.0;

      // Apply rotation through shape_rotation_angle about the z-axis in the reference configuration.
      const matrix Rshape = (quaternion(std::cos(0.5*shape_rotation_angle), 0.0, 0.0, std::sin(0.5*shape_rotation_angle))).rot_mat();

    #endif

    #if FIT_TO_DATA_BEAT

      find_fitted_shape_s();

    #elif BUILD_A_BEAT

      matrix t1(3,1), t2(3,1), k1(3,1), k2(3,1);

      build_a_beat_tangent(t1, 0.0);
      t1 = R*t1;

      build_a_beat_tangent_phase_deriv(k1, 0.0);
      k1 = R*k1;

      #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

        t1 = R*Rshape*t1;
        k1 = R*Rshape*k1;

      #else

        t1 = R*t1;
        k1 = R*k1;

      #endif

    #endif

    for (int n = 1; n < NSEG; n++){

      #if FIT_TO_DATA_BEAT

        #if BICILIA
          int index_in_pair = floor(n/NSEG_PER_CILIA);
          Real z_dis = index_in_pair*2.2*RSEG;
          Real s_to_use_n = index_in_pair ? s_to_use2[n%NSEG_PER_CILIA] : s_to_use[n%NSEG_PER_CILIA];
          Real phase_this_cilia = index_in_pair ? phase2 : phase;
        #elif BICILIA_LONGT
          int index_in_pair = floor(n/NSEG_PER_CILIA);
          Real z_dis = index_in_pair*2.2*RSEG;
          Real s_to_use_n = index_in_pair ? s_to_use2[n%NSEG_PER_CILIA] : s_to_use[n%NSEG_PER_CILIA];
          Real phase_this_cilia = index_in_pair ? phase2 : phase;
        #else
          Real s_to_use_n = s_to_use[n];
          int index_in_pair = 0;
          Real z_dis = 0.0;
          Real phase_this_cilia = phase;
        #endif;

        #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

          matrix ref = FIL_LENGTH*Rshape*fitted_shape(s_to_use_n, phase_this_cilia, z_dis/FIL_LENGTH);

          const matrix diff = R*ref;

          // Turn ref into cross(e_z, ref)
          ref(2) = -ref(1);
          ref(1) = ref(0);
          ref(0) = ref(2);
          ref(2) = 0.0;

          ref = R*ref;

          vel_dir_angle[3*n] = ref(0);
          vel_dir_angle[3*n + 1] = ref(1);
          vel_dir_angle[3*n + 2] = ref(2);

        #else

          matrix diff = FIL_LENGTH*R*fitted_shape(s_to_use_n, phase_this_cilia, z_dis/FIL_LENGTH);

        #endif

        segments[n].x[0] = segments[0].x[0] + diff(0);
        segments[n].x[1] = segments[0].x[1] + diff(1);
        segments[n].x[2] = segments[0].x[2] + diff(2);

        // std::cout << id << " "<<n << " " << segments[0].x[0] << "  "<< s_to_use_n <<" "<<phase_this_cilia << std::endl;

        #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

          const matrix dir = FIL_LENGTH*R*Rshape*fitted_shape_velocity_direction(s_to_use_n, phase_this_cilia);

        #else

          const matrix dir = FIL_LENGTH*R*fitted_shape_velocity_direction(s_to_use_n, phase_this_cilia);

        #endif

        vel_dir_phase[3*n] = dir(0);
        vel_dir_phase[3*n + 1] = dir(1);
        vel_dir_phase[3*n + 2] = dir(2);

      #elif BUILD_A_BEAT

        // Shape
        build_a_beat_tangent(t2, Real(n)/Real(NSEG - 1));

        #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

          t2 = R*Rshape*t2;

        #else

          t2 = R*t2;

        #endif

        segments[n].x[0] = segments[n-1].x[0] + 0.5*DL*(t1(0) + t2(0));
        segments[n].x[1] = segments[n-1].x[1] + 0.5*DL*(t1(1) + t2(1));
        segments[n].x[2] = segments[n-1].x[2] + 0.5*DL*(t1(2) + t2(2));

        t1 = t2;

        // Velocity direction
        build_a_beat_tangent_phase_deriv(k2, Real(n)/Real(NSEG - 1));

        #if (DYNAMIC_SHAPE_ROTATION || WRITE_GENERALISED_FORCES)

          k2 = R*Rshape*k2;

          matrix ref(3,1);
          ref(0) = segments[n].x[0] - segments[0].x[0];
          ref(1) = segments[n].x[1] - segments[0].x[1];
          ref(2) = segments[n].x[2] - segments[0].x[2];

          ref = Rshape*ref;

          // Turn ref into cross(e_z, ref)
          ref(2) = -ref(1);
          ref(1) = ref(0);
          ref(0) = ref(2);
          ref(2) = 0.0;

          ref = R*ref;

          vel_dir_angle[3*n] = ref(0);
          vel_dir_angle[3*n + 1] = ref(1);
          vel_dir_angle[3*n + 2] = ref(2);

        #else

          k2 = R*k2;

        #endif

        vel_dir_phase[3*n] = vel_dir_phase[3*(n-1)] + 0.5*DL*(k1(0) + k2(0));
        vel_dir_phase[3*n + 1] = vel_dir_phase[3*(n-1) + 1] + 0.5*DL*(k1(1) + k2(1));
        vel_dir_phase[3*n + 2] = vel_dir_phase[3*(n-1) + 2] + 0.5*DL*(k1(2) + k2(2));

        k1 = k2;

      #endif

    }

  #else

    if (nt == 1){

      for (int n = 0; n < 3*(NSEG-1); n++){

        lambdam1[n] = lambda[n];

      }

      tether_lambdam1[0] = tether_lambda[0];
      tether_lambdam1[1] = tether_lambda[1];
      tether_lambdam1[2] = tether_lambda[2];

      #if !(GEOMETRIC_CILIA || NO_CILIA_SQUIRMER)

        clamp_lambdam1[0] = clamp_lambda[0];
        clamp_lambdam1[1] = clamp_lambda[1];
        clamp_lambdam1[2] = clamp_lambda[2];

      #endif

    } else if (nt == 2){

      for (int n = 0; n < 3*(NSEG-1); n++){

        lambdam2[n] = lambdam1[n];
        lambdam1[n] = lambda[n];
        lambda[n] = 2.0*lambdam1[n] - lambdam2[n];

      }

      tether_lambdam2[0] = tether_lambdam1[0];
      tether_lambdam1[0] = tether_lambda[0];
      tether_lambda[0] = 2.0*tether_lambdam1[0] - tether_lambdam2[0];

      tether_lambdam2[1] = tether_lambdam1[1];
      tether_lambdam1[1] = tether_lambda[1];
      tether_lambda[1] = 2.0*tether_lambdam1[1] - tether_lambdam2[1];

      tether_lambdam2[2] = tether_lambdam1[2];
      tether_lambdam1[2] = tether_lambda[2];
      tether_lambda[2] = 2.0*tether_lambdam1[2] - tether_lambdam2[2];

      #if !(GEOMETRIC_CILIA || NO_CILIA_SQUIRMER)

        clamp_lambdam2[0] = clamp_lambdam1[0];
        clamp_lambdam1[0] = clamp_lambda[0];
        clamp_lambda[0] = 2.0*clamp_lambdam1[0] - clamp_lambdam2[0];

        clamp_lambdam2[1] = clamp_lambdam1[1];
        clamp_lambdam1[1] = clamp_lambda[1];
        clamp_lambda[1] = 2.0*clamp_lambdam1[1] - clamp_lambdam2[1];

        clamp_lambdam2[2] = clamp_lambdam1[2];
        clamp_lambdam1[2] = clamp_lambda[2];
        clamp_lambda[2] = 2.0*clamp_lambdam1[2] - clamp_lambdam2[2];

      #endif

    } else if (nt > 2){

      for (int n = 0; n < 3*(NSEG-1); n++){

        const Real temp = 3.0*(lambda[n] - lambdam1[n]) + lambdam2[n];

        lambdam2[n] = lambdam1[n];
        lambdam1[n] = lambda[n];
        lambda[n] = temp;

      }

      Real temp = 3.0*(tether_lambda[0] - tether_lambdam1[0]) + tether_lambdam2[0];
      tether_lambdam2[0] = tether_lambdam1[0];
      tether_lambdam1[0] = tether_lambda[0];
      tether_lambda[0] = temp;

      temp = 3.0*(tether_lambda[1] - tether_lambdam1[1]) + tether_lambdam2[1];
      tether_lambdam2[1] = tether_lambdam1[1];
      tether_lambdam1[1] = tether_lambda[1];
      tether_lambda[1] = temp;

      temp = 3.0*(tether_lambda[2] - tether_lambdam1[2]) + tether_lambdam2[2];
      tether_lambdam2[2] = tether_lambdam1[2];
      tether_lambdam1[2] = tether_lambda[2];
      tether_lambda[2] = temp;

      #if !(GEOMETRIC_CILIA || NO_CILIA_SQUIRMER)

        temp = 3.0*(clamp_lambda[0] - clamp_lambdam1[0]) + clamp_lambdam2[0];
        clamp_lambdam2[0] = clamp_lambdam1[0];
        clamp_lambdam1[0] = clamp_lambda[0];
        clamp_lambda[0] = temp;

        temp = 3.0*(clamp_lambda[1] - clamp_lambdam1[1]) + clamp_lambdam2[1];
        clamp_lambdam2[1] = clamp_lambdam1[1];
        clamp_lambdam1[1] = clamp_lambda[1];
        clamp_lambda[1] = temp;

        temp = 3.0*(clamp_lambda[2] - clamp_lambdam1[2]) + clamp_lambdam2[2];
        clamp_lambdam2[2] = clamp_lambdam1[2];
        clamp_lambdam1[2] = clamp_lambda[2];
        clamp_lambda[2] = temp;

      #endif

    }

  #endif

}

void filament::end_of_step(const int nt){

  #if GEOMETRIC_CILIA

    if (just_switched_from_fast_to_slow){

      int tid = nt - step_at_start_of_transition;

      base_torque_magnitude_factor = 1.0 + tid*((-1.0/DRIVING_TORQUE_MAGNITUDE_RATIO) - 1.0)/TRANSITION_STEPS;

      if (tid == TRANSITION_STEPS){

        just_switched_from_fast_to_slow = false;

      }

    } else if (just_switched_from_slow_to_fast){

      int tid = nt - step_at_start_of_transition;

      base_torque_magnitude_factor = (-1.0/DRIVING_TORQUE_MAGNITUDE_RATIO) + tid*(1.0 + (1.0/DRIVING_TORQUE_MAGNITUDE_RATIO))/TRANSITION_STEPS;

      if (tid == TRANSITION_STEPS){

        just_switched_from_slow_to_fast = false;

      }

    } else if (find_my_angle() > CRITICAL_ANGLE){ // The angle is defined such that it increases in the fast stroke.

      step_at_start_of_transition = nt;

      just_switched_from_fast_to_slow = true;

    } else if (find_my_angle() < -CRITICAL_ANGLE){

      step_at_start_of_transition = nt;

      just_switched_from_slow_to_fast = true;

    }

  #endif

}

#if GEOMETRIC_CILIA

  Real filament::find_my_angle(){

    /*const vec3 vertical = body_q.rotate(my_vertical);
    const vec3 positive_angle_direction = body_q.rotate(fast_beating_direction);
    const vec3 tangent = segments[0].tangent();
    const Real x_component = dot(tangent, positive_angle_direction);
    const Real z_component = dot(tangent, vertical);

    return atan(x_component/z_component);*/

    return 0.0;

  }

#endif

matrix filament::body_frame_moment(const int link_id){

  quaternion midq = midpoint_quaternion(segments[link_id].q, segments[link_id+1].q);
  midq.conj_in_place();
  midq = 2.0*(midq * (segments[link_id+1].q - segments[link_id].q)/DL);

  matrix M(3,1);

  M(0) = KT*(midq.vector_part[0] - strain_twist[0]);
  M(1) = KB*(midq.vector_part[1] - strain_twist[1]);
  M(2) = KB*(midq.vector_part[2] - strain_twist[2]);

  return M;

}

void filament::internal_forces_and_torques(const int nt){

  // Forces and torques are stored in a global array.
  for (int n = 0; n < 6*NSEG; n++){

    f[n] = 0.0;

  }

  Real t[3];
  matrix R(3,3);
  quaternion qmid;

  for (int i = 0; i < NSEG-1; i++){

    // Inex. constraint forces and torques
    const Real *const lam = &lambda[3*i];

    f[6*i] -= lam[0];
    f[6*i + 1] -= lam[1];
    f[6*i + 2] -= lam[2];

    f[6*(i+1)] += lam[0];
    f[6*(i+1) + 1] += lam[1];
    f[6*(i+1) + 2] += lam[2];

    segments[i].tangent(t);
    f[6*i + 3] -= 0.5*DL*(t[1]*lam[2] - t[2]*lam[1]);
    f[6*i + 4] -= 0.5*DL*(t[2]*lam[0] - t[0]*lam[2]);
    f[6*i + 5] -= 0.5*DL*(t[0]*lam[1] - t[1]*lam[0]);

    segments[i+1].tangent(t);
    f[6*(i+1) + 3] -= 0.5*DL*(t[1]*lam[2] - t[2]*lam[1]);
    f[6*(i+1) + 4] -= 0.5*DL*(t[2]*lam[0] - t[0]*lam[2]);
    f[6*(i+1) + 5] -= 0.5*DL*(t[0]*lam[1] - t[1]*lam[0]);

    // Elastic torques
    midpoint_quaternion(qmid, segments[i].q, segments[i+1].q);
    qmid.rot_mat(R);
    const matrix M = R*body_frame_moment(i);

    f[6*i + 3] += M(0);
    f[6*i + 4] += M(1);
    f[6*i + 5] += M(2);

    f[6*(i+1) + 3] -= M(0);
    f[6*(i+1) + 4] -= M(1);
    f[6*(i+1) + 5] -= M(2);

  }

  // Tethering force and driving torque
  f[0] += tether_lambda[0];
  f[1] += tether_lambda[1];
  f[2] += tether_lambda[2];

  #if GEOMETRIC_CILIA

    body_q.matrix(R);
    const vec3 driving_torque = base_torque_magnitude_factor*DEFAULT_BASE_TORQUE_MAGNITUDE*R*cross(my_vertical, fast_beating_direction);

    f[3] += driving_torque(0);
    f[4] += driving_torque(1);
    f[5] += driving_torque(2);

  #elif INSTABILITY_CILIA

    segments[NSEG-1].tangent(t);

    f[6*(NSEG-1)] -= END_FORCE_MAGNITUDE*t[0];
    f[6*(NSEG-1) + 1] -= END_FORCE_MAGNITUDE*t[1];
    f[6*(NSEG-1) + 2] -= END_FORCE_MAGNITUDE*t[2];

    #if !READ_INITIAL_CONDITIONS_FROM_BACKUP

      if (nt == 0){

        // Initial perturbation
        int i = 4;
        f[6*i] += perturbation1[0];
        f[6*i + 1] += perturbation1[1];
        f[6*i + 2] += perturbation1[2];

        i = 12;
        f[6*i] += perturbation2[0];
        f[6*i + 1] += perturbation2[1];
        f[6*i + 2] += perturbation2[2];

      }

    #endif

    Real clamping_torque[3];
    dexpinv_transpose(clamping_torque, segments[0].u, clamp_lambda);
    f[3] += clamping_torque[0];
    f[4] += clamping_torque[1];
    f[5] += clamping_torque[2];

  #elif CONSTANT_BASE_ROTATION

    Real clamping_torque[3];
    dexpinv_transpose(clamping_torque, segments[0].u, clamp_lambda);
    f[3] += clamping_torque[0];
    f[4] += clamping_torque[1];
    f[5] += clamping_torque[2];

  #endif

}

matrix filament::jacobian_lie_algebra_block(const int nt){

  matrix E(3*NSEG, 3*NSEG);
  E.zero();

  const Real time_step_fac = (nt < NUM_EULER_STEPS) ? DT : 2.0*DT/3.0;

  // We deal with the end particles seperately:

  // n = 0
  const Real Tfac = 1.0/(8.0*PI*MU*RSEG*RSEG*RSEG);

  Real t[3], n[3], b[3], tp1[3], np1[3], bp1[3];

  segments[0].tangent(t);
  segments[0].normal(n);
  segments[0].binormal(b);

  segments[1].tangent(tp1);
  segments[1].normal(np1);
  segments[1].binormal(bp1);

  // Diagonal block
  matrix ConstraintPart = -0.5*DL*rcross(&lambda[0])*rcross(t);

  quaternion qph = midpoint_quaternion(segments[0].q, segments[1].q); // ph = plus_a_half
  matrix Kph = body_frame_moment(0);
  matrix KphLieDerivative = body_frame_moment_lie_derivative(segments[0].q, segments[1].q, false);
  matrix ElasticPart = 0.5*(Kph(0)*rcross(t) + Kph(1)*rcross(n) + Kph(2)*rcross(b));

  matrix R = qph.rot_mat();

  ElasticPart += R*KphLieDerivative;

  matrix dOmega = Tfac*(ElasticPart + ConstraintPart);

  matrix LieMat = -rcross(segments[0].u); // = transpose of rcross
  Real approx_Omega[3] = {Tfac*f[3], Tfac*f[4], Tfac*f[5]};
  matrix OmegaMat = rcross(approx_Omega);
  matrix u_vec(3, 1, segments[0].u);
  matrix Block = -time_step_fac*(dOmega - 0.5*(OmegaMat + LieMat*dOmega) + (rcross(OmegaMat*u_vec) + LieMat*OmegaMat + LieMat*LieMat*dOmega)/12.0);

  E.set_block(0, 0, 3, 3, Block);

  // Off-diagonal block
  KphLieDerivative = body_frame_moment_lie_derivative(segments[0].q, segments[1].q, true);
  ElasticPart = 0.5*(Kph(0)*rcross(tp1) + Kph(1)*rcross(np1) + Kph(2)*rcross(bp1));
  ElasticPart += R*KphLieDerivative;
  dOmega = Tfac*ElasticPart;
  Block = -time_step_fac*(dOmega - 0.5*LieMat*dOmega + LieMat*LieMat*dOmega/12.0);
  E.set_block(0, 3, 3, 3, Block);

  // End particle
  segments[NSEG-1].tangent(t);
  segments[NSEG-1].normal(n);
  segments[NSEG-1].binormal(b);

  Real tm1[3], nm1[3], bm1[3];

  segments[NSEG-2].tangent(tm1);
  segments[NSEG-2].normal(nm1);
  segments[NSEG-2].binormal(bm1);

  // Diagonal block
  ConstraintPart = -0.5*DL*rcross(&lambda[3*(NSEG-2)])*rcross(t);

  quaternion qmh = midpoint_quaternion(segments[NSEG-2].q, segments[NSEG-1].q);
  matrix Kmh = body_frame_moment(NSEG-2); // mh = minus_a_half
  matrix KmhLieDerivative = body_frame_moment_lie_derivative(segments[NSEG-2].q, segments[NSEG-1].q, true);
  ElasticPart = -0.5*(Kmh(0)*rcross(t) + Kmh(1)*rcross(n) + Kmh(2)*rcross(b));

  qmh.rot_mat(R);

  ElasticPart -= R*KmhLieDerivative;

  dOmega = Tfac*(ElasticPart + ConstraintPart);
  LieMat = -rcross(segments[NSEG-1].u);
  approx_Omega[0] = Tfac*f[6*(NSEG-1) + 3];
  approx_Omega[1] = Tfac*f[6*(NSEG-1) + 4];
  approx_Omega[2] = Tfac*f[6*(NSEG-1) + 5];
  rcross(OmegaMat, approx_Omega);
  u_vec(0) = segments[NSEG-1].u[0];
  u_vec(1) = segments[NSEG-1].u[1];
  u_vec(2) = segments[NSEG-1].u[2];
  Block = -time_step_fac*(dOmega - 0.5*(OmegaMat + LieMat*dOmega) +
                 (rcross(OmegaMat*u_vec) + LieMat*OmegaMat + LieMat*LieMat*dOmega)/12.0);
  E.set_block(3*NSEG-3, 3*NSEG-3, 3, 3, Block);

  // Off-diagonal block
  KmhLieDerivative = body_frame_moment_lie_derivative(segments[NSEG-2].q, segments[NSEG-1].q, false);
  ElasticPart = -0.5*(Kmh(0)*rcross(tm1) + Kmh(1)*rcross(nm1) + Kmh(2)*rcross(bm1));
  ElasticPart -= R*KmhLieDerivative;
  dOmega = Tfac*ElasticPart;
  Block = -time_step_fac*(dOmega - 0.5*LieMat*dOmega + LieMat*LieMat*dOmega/12.0);
  E.set_block(3*NSEG-3, 3*NSEG-6, 3, 3, Block);

  // Now we loop over the interior particles, which experience elastic and constraint
  // torques on both sides and thus have the most general form to their entries.
  for (int k = 1; k < NSEG-1; k++) {

    segments[k].tangent(t);
    segments[k].normal(n);
    segments[k].binormal(b);

    segments[k-1].tangent(tm1);
    segments[k-1].normal(nm1);
    segments[k-1].binormal(bm1);

    segments[k+1].tangent(tp1);
    segments[k+1].normal(np1);
    segments[k+1].binormal(bp1);

    LieMat = -rcross(segments[k].u);
    approx_Omega[0] = Tfac*f[6*k + 3];
    approx_Omega[1] = Tfac*f[6*k + 4];
    approx_Omega[2] = Tfac*f[6*k + 5];
    rcross(OmegaMat, approx_Omega);

    Kph = body_frame_moment(k);
    Kmh = body_frame_moment(k-1);

    midpoint_quaternion(qph, segments[k].q, segments[k+1].q);
    midpoint_quaternion(qmh, segments[k-1].q, segments[k].q);

    // Diagonal block
    const Real lamvec[3] = {lambda[3*(k-1)] + lambda[3*k], lambda[3*(k-1) + 1] + lambda[3*k + 1], lambda[3*(k-1) + 2] + lambda[3*k + 2]};
    ConstraintPart = -0.5*DL*rcross(lamvec)*rcross(t);

    KphLieDerivative = body_frame_moment_lie_derivative(segments[k].q, segments[k+1].q, false);
    ElasticPart = 0.5*(Kph(0)*rcross(t) + Kph(1)*rcross(n) + Kph(2)*rcross(b));

    qph.rot_mat(R);

    ElasticPart += R*KphLieDerivative;

    KmhLieDerivative = body_frame_moment_lie_derivative(segments[k-1].q, segments[k].q, true);
    ElasticPart -= 0.5*(Kmh(0)*rcross(t) + Kmh(1)*rcross(n) + Kmh(2)*rcross(b));

    qmh.rot_mat(R);

    ElasticPart -= R*KmhLieDerivative;

    dOmega = Tfac*(ConstraintPart + ElasticPart);
    u_vec(0) = segments[k].u[0];
    u_vec(1) = segments[k].u[1];
    u_vec(2) = segments[k].u[2];
    Block = -time_step_fac*(dOmega - 0.5*(OmegaMat + LieMat*dOmega) + (rcross(OmegaMat*u_vec) + LieMat*OmegaMat + LieMat*LieMat*dOmega)/12.0);
    E.set_block(3*k, 3*k, 3, 3, Block);

    // Next, the left off-diagonal block
    KmhLieDerivative = body_frame_moment_lie_derivative(segments[k-1].q, segments[k].q, false);
    ElasticPart = -0.5*(Kmh(0)*rcross(tm1) + Kmh(1)*rcross(nm1) + Kmh(2)*rcross(bm1));

    ElasticPart -= R*KmhLieDerivative;

    dOmega = Tfac*ElasticPart;
    Block = -time_step_fac*(dOmega - 0.5*LieMat*dOmega + LieMat*LieMat*dOmega/12.0);
    E.set_block(3*k, 3*k - 3, 3, 3, Block);

    // Finally, the right off-diagonal block
    KphLieDerivative = body_frame_moment_lie_derivative(segments[k].q,segments[k+1].q, true);
    ElasticPart = 0.5*(Kph(0)*rcross(tp1) + Kph(1)*rcross(np1) + Kph(2)*rcross(bp1));

    qph.rot_mat(R);

    ElasticPart += R*KphLieDerivative;

    dOmega = Tfac*ElasticPart;
    Block = -time_step_fac*(dOmega - 0.5*LieMat*dOmega + LieMat*LieMat*dOmega/12.0);
    E.set_block(3*k, 3*k + 3, 3, 3, Block);

  }

  // All that remains is to add the identity
  for (int k = 0; k < 3*NSEG; k++){

    E(k,k) += 1.0;

  }

  return E;

}

void filament::invert_approx_jacobian(const int nt){

  // Introduce an alias for the inverse Jacobian class member.
  // We will invert the approx. Jacobian in-place, leaving the inverse stored in inverse_jacobian, as we expect.
  matrix& Japprox = inverse_jacobian;
  Japprox.zero();

  // Begin with the dependence of the position equations on the Lagrange
  // multipliers (both those relating to inextensibility and the tethering constraint).
  Real fac = (nt < NUM_EULER_STEPS) ? -DT/(6.0*PI*MU*RSEG) : -DT/(9.0*PI*MU*RSEG);

  for (int i = 0; i < 3*NSEG; i++) {

    Japprox(i,i) = fac;

    if (i < 3*NSEG-3){

      Japprox(i,i+3) = -fac;

    }

  }

  // Now we include the dependence of these same equations on the Lie algebra
  // elements.
  Real t[3];
  matrix temp(3,3);

  #if GEOMETRIC_CILIA

    segments[0].tangent(t);
    rcross(temp, t);
    temp *= 0.5*DL;

    for (int i = 1; i < NSEG; i++){

      Japprox.set_block(3*i, 3*NSEG, 3, 3, temp);

    }

  #endif

  for (int j = 1; j < NSEG; j++){

    int jpos = 3*(NSEG+j);

    segments[j].tangent(t);
    rcross(temp, t);
    temp *= DL;

    Japprox.set_block(3*j, jpos, 3, 3, 0.5*temp);

    for (int i = j+1; i < NSEG; i++){

      Japprox.set_block(3*i, jpos, 3, 3, temp);

    }

  }

  #if INSTABILITY_CILIA

    segments[NSEG-1].tangent(t);

    Japprox.subtract_from_block(3*(NSEG-1), 3*(2*NSEG-1), 3, 3, fac*END_FORCE_MAGNITUDE*rcross(t));

  #endif

  // Next, we include the dependence of the Lie algebra update equations on
  // the inextensibility and tethering Lagrange multipliers.
  fac = (nt < NUM_EULER_STEPS) ? DL*DT/(16.0*PI*MU*RSEG*RSEG*RSEG) : DL*DT/(24.0*PI*MU*RSEG*RSEG*RSEG);

  for (int j = 1; j < NSEG; j++){

    int jpos = 3*j;
    int ipos = 3*(NSEG + j - 1);

    segments[j-1].tangent(t);

    matrix umat = rcross(segments[j-1].u);
    matrix tmat = -rcross(t); // = transpose of rcross(t)

    Japprox.set_block(ipos, jpos, 3, 3, fac*(tmat + 0.5*umat*tmat + umat*umat*tmat/12.0));

    ipos += 3;

    segments[j].tangent(t);

    rcross(umat, segments[j].u);
    rcross(tmat, t);
    tmat *= -1.0;

    Japprox.set_block(ipos, jpos, 3, 3, fac*(tmat + 0.5*umat*tmat + umat*umat*tmat/12.0));

  }

  // The final sub-block to construct before we are ready to invert the matrix
  // is the one encoding the dependence of the Lie algebra equations on the
  // Lie algebra elements, or on the Lagrange multiplier associated with the
  // imposed rotation and the Lie algebra elements.

  // Since it's a very messy expression, it's computed in a subfunction.
  Japprox.set_block(3*NSEG, 3*NSEG, 3*NSEG, 3*NSEG, jacobian_lie_algebra_block(nt));

  // If we're constraining the first frame, we adjust the first 3 columns of this
  // sub-block.

  #if (CILIA_TYPE != 1)

    fac = (nt < NUM_EULER_STEPS) ? -DT/(8.0*PI*MU*RSEG*RSEG*RSEG) : -DT/(12.0*PI*MU*RSEG*RSEG*RSEG);

    matrix umat = rcross(segments[0].u);

    matrix block = 0.5*umat + umat*umat/12.0;
    block(0,0) += 1.0;
    block(1,1) += 1.0;
    block(2,2) += 1.0;
    matrix block_transpose = -0.5*umat + umat*umat/12.0;
    block_transpose(0,0) += 1.0;
    block_transpose(1,1) += 1.0;
    block_transpose(2,2) += 1.0;

    Japprox.get_block(3*NSEG, 3*NSEG, 3, 3, elastic_clamping_block1);
    Japprox.get_block(3*NSEG + 3, 3*NSEG, 3, 3, elastic_clamping_block2);

    Japprox.set_block(3*NSEG, 3*NSEG, 3, 3, fac*block*block_transpose);
    Japprox.set_block(3*NSEG + 3, 3*NSEG, 3, 3, 0.0);

  #endif

  // Finally, we compute and store the inverse of the initial approximate Jacobian.
  Japprox.invert();

}

void filament::update(const Real *const u){

  tether_lambda[0] += u[0];
  tether_lambda[1] += u[1];
  tether_lambda[2] += u[2];

  for (int i = 1; i < NSEG; i++){

    Real *const lam = &lambda[3*(i-1)];

    lam[0] += u[3*i];
    lam[1] += u[3*i + 1];
    lam[2] += u[3*i + 2];

    const Real *const seg_update = &u[3*(NSEG+i)];
    segments[i].update(seg_update);

  }

  #if !(GEOMETRIC_CILIA || PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

    clamp_lambda[0] += u[3*NSEG];
    clamp_lambda[1] += u[3*NSEG + 1];
    clamp_lambda[2] += u[3*NSEG + 2];

  #elif GEOMETRIC_CILIA

    const Real *const seg_update = &u[3*NSEG];
    segments[0].update(seg_update);

  #endif

  robot_arm();

}

void filament::write_data(std::ofstream& data_file) const {

  for (int n = 0; n < NSEG; n++){

    #if PRESCRIBED_CILIA

      data_file << segments[n].x[0] << " " << segments[n].x[1] << " " << segments[n].x[2] << " ";

    #else

      data_file << segments[n].q.scalar_part << " " << segments[n].q.vector_part[0] << " " << segments[n].q.vector_part[1] << " " << segments[n].q.vector_part[2] << " ";

    #endif

  }

}

void filament::write_backup(std::ofstream& data_file) const {

  #if PRESCRIBED_CILIA

    data_file << phase << " ";
    data_file << phase_dot << " ";
    data_file << omega0 << " ";

    data_file << body_qm1.scalar_part << " ";
    data_file << body_qm1.vector_part[0] << " ";
    data_file << body_qm1.vector_part[1] << " ";
    data_file << body_qm1.vector_part[2] << " ";

    data_file << body_q.scalar_part << " ";
    data_file << body_q.vector_part[0] << " ";
    data_file << body_q.vector_part[1] << " ";
    data_file << body_q.vector_part[2] << " ";

    #if DYNAMIC_SHAPE_ROTATION

      data_file << shape_rotation_angle << " ";
      data_file << shape_rotation_angle_dot << " ";

    #else

      // If we don't store 0 in this case, the code would go haywire if we tried to start a DYNAMIC_SHAPE_ROTATION=true sim from a DYNAMIC_SHAPE_ROTATION=false backup.
      // Note that there's nothing wrong with doing this kind of hold-and-release resume, but the converse, where we hold fixed an angle which was previously free to
      // evolve, doesn't make sense because of how the angle is always 0 in the reference problem.
      data_file << 0.0 << " ";
      data_file << 0.0 << " ";

    #endif

  #elif NO_CILIA_SQUIRMER

  // There are no filaments so we don't need this to do anything, but this is the easiest way of getting the compiler not to have a go at me...
  // This is also why several of the other filament methods have options involving the macro NO_CILIA_SQUIRMER.

  #else

    for (int n = 0; n < NSEG; n++){

      data_file << segments[n].q.scalar_part << " " << segments[n].q.vector_part[0] << " " << segments[n].q.vector_part[1] << " " << segments[n].q.vector_part[2] << " ";

      data_file << segments[n].x[0] << " " << segments[n].x[1] << " " << segments[n].x[2] << " ";
      data_file << segments[n].xm1[0] << " " << segments[n].xm1[1] << " " << segments[n].xm1[2] << " ";
      data_file << segments[n].xm2[0] << " " << segments[n].xm2[1] << " " << segments[n].xm2[2] << " ";

      data_file << segments[n].u[0] << " " << segments[n].u[1] << " " << segments[n].u[2] << " ";
      data_file << segments[n].um1[0] << " " << segments[n].um1[1] << " " << segments[n].um1[2] << " ";

    }

    data_file << tether_lambda[0] << " " << tether_lambda[1] << " " << tether_lambda[2] << " ";
    data_file << tether_lambdam1[0] << " " << tether_lambdam1[1] << " " << tether_lambdam1[2] << " ";
    data_file << tether_lambdam2[0] << " " << tether_lambdam2[1] << " " << tether_lambdam2[2] << " ";

    data_file << clamp_lambda[0] << " " << clamp_lambda[1] << " " << clamp_lambda[2] << " ";
    data_file << clamp_lambdam1[0] << " " << clamp_lambdam1[1] << " " << clamp_lambdam1[2] << " ";
    data_file << clamp_lambdam2[0] << " " << clamp_lambdam2[1] << " " << clamp_lambdam2[2] << " ";

    for (int n = 0; n < NSEG-1; n++){

      data_file << lambda[3*n] << " " << lambda[3*n + 1] << " " << lambda[3*n + 2] << " ";
      data_file << lambdam1[3*n] << " " << lambdam1[3*n + 1] << " " << lambdam1[3*n + 2] << " ";
      data_file << lambdam2[3*n] << " " << lambdam2[3*n + 1] << " " << lambdam2[3*n + 2] << " ";

    }

  #endif

}











#if PRESCRIBED_CILIA

  #include <sstream>

  std::string reference_file_name(const char *const file_type){

    #if FULFORD_AND_BLAKE_BEAT

      return std::string("input/forcing/fulford_and_blake_reference_") + std::string(file_type) + "_NSEG=" + std::to_string(NSEG) + "_SEP=" + std::to_string(SEG_SEP) + std::string(".dat");

    #elif FULFORD_AND_BLAKE_BEAT_NO_WALL

      return std::string("input/forcing/nowall_fulford_and_blake_reference_") + std::string(file_type) + "_NSEG=" + std::to_string(NSEG) + "_SEP=" + std::to_string(SEG_SEP) + std::string(".dat");

    #elif CORAL_LARVAE_BEAT

      return std::string("coral_larvae_reference_") + std::string(file_type) + std::string(".dat");

    #elif VOLVOX_BEAT 

      return std::string("input/forcing/volvox_reference_") + std::string(file_type) + "_NSEG=" + std::to_string(NSEG) + "_SEP=" + std::to_string(SEG_SEP) + std::string(".dat");
    
    #elif FULFORD_AND_BLAKE_BEAT_ORIGINAL

      return std::string("input/forcing/fulford_and_blake_original_reference_") + std::string(file_type) + "_NSEG=" + std::to_string(NSEG) + "_SEP=" + std::to_string(SEG_SEP) + std::string(".dat");

    #elif BUILD_A_BEAT

      // Using std::stringstream rather than std::string allows us to set the precision.
      std::stringstream filename_stringstream;
      filename_stringstream.precision(5);
      filename_stringstream << std::fixed;

      filename_stringstream << "build_a_beat_";
      filename_stringstream << SCALED_BEAT_AMPLITUDE;
      filename_stringstream << "_";
      filename_stringstream << RECOVERY_STROKE_WINDOW_LENGTH;
      filename_stringstream << "_";
      filename_stringstream << EFFECTIVE_STROKE_LENGTH;
      filename_stringstream << "_";
      filename_stringstream << ZERO_VELOCITY_AVOIDANCE_LENGTH;
      filename_stringstream << "_reference_";
      filename_stringstream << file_type;
      filename_stringstream << ".dat";

      return filename_stringstream.str();

    #elif BICILIA

      return std::string("input/forcing/bicilia_reference_") + std::string(file_type) + "_NSEG=" + std::to_string(NSEG) + "_SEP=" + std::to_string(SEG_SEP) + "_PAIR_DP=" + std::to_string(PAIR_DP)  + std::string(".dat");
    
    #elif BICILIA_LONGT

      return std::string("input/forcing/bicilia_longt_reference_") + std::string(file_type) + "_NSEG=" + std::to_string(NSEG) + "_SEP=" + std::to_string(SEG_SEP) + "_PAIR_DP=" + std::to_string(PAIR_DP)  + std::string(".dat");
    

    #endif

  }

  std::string reference_phase_generalised_force_file_name(){

    return reference_file_name("phase_generalised_forces");

  }

  std::string reference_angle_generalised_force_file_name(){
    
    return reference_file_name("angle_generalised_forces");

  }

  std::string reference_s_values_file_name(){

    return reference_file_name("s_values");

  }

#endif
