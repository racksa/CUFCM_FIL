// swimmer.cpp

#include <cmath>
#include <string>
#include <algorithm>
#include "swimmer.hpp"
#include "seeding.hpp"
#include "util.hpp"
#include "../general/matrix.hpp"
#include "../../config.hpp"

swimmer::~swimmer(){}

swimmer::swimmer(){}

void swimmer::initial_setup(const int id, const Real *const data_from_file, Real *const x_segs_address, Real *const f_segs_address, Real *const f_blobs_address){

  filament_references = std::vector<Real>(3*NFIL);
  polar_dir_refs = std::vector<Real>(3*NFIL);
  azi_dir_refs = std::vector<Real>(3*NFIL);
  normal_refs = std::vector<Real>(3*NFIL);

  filaments = std::vector<filament>(NFIL);

  f = matrix(6,1);
  f.zero();

  #if PRESCRIBED_CILIA

    KTMinvK_inv = matrix(6,6);

  #else

    #if !INFINITE_PLANE_WALL

      schur_mat_inv = matrix(6,6);
      jacobian_B_blocks = std::vector<matrix>(NFIL);

    #endif

  #endif

  #if READ_INITIAL_CONDITIONS_FROM_BACKUP

    #if PRESCRIBED_CILIA

      const int data_per_fil = 13;

    #else

      const int data_per_fil = 9 + 28*NSEG;

    #endif

    body.initial_setup(id, f_blobs_address, &data_from_file[NFIL*data_per_fil]);

  #else
    body.initial_setup(id, f_blobs_address, data_from_file);

  #endif

  #if INFINITE_PLANE_WALL

    #if (DEFINED_BUT_EMPTY(FIL_LATTICE_X_NUM) && DEFINED_BUT_EMPTY(FIL_LATTICE_Y_NUM))

      // Neither value is given, so we attempt to make a regular lattice.
      #if RECTANGULAR_SEEDING

        const int fil_grid_dim_x = int(sqrt(Real(NFIL)));
        const int fil_grid_dim_y = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x))));

      #elif HEXAGONAL_SEEDING

        const int fil_grid_dim_x = std::round(0.25*(3.0 - std::sqrt(3.0) + std::sqrt(4.0 - 2.0*std::sqrt(3.0) + 8.0*std::sqrt(3.0)*NFIL)));
        const int fil_grid_dim_y = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x-1)))); // Ensure we have enough rows even if all rows were of the shorter type.
      
      #elif FCM_LATTICE_SEEDING or FCM_RECTANGULAR_SEEDING

        const int fil_grid_dim_x = int(cbrt(Real(NFIL)));
        const int fil_grid_dim_y = std::max<int>(1, int(sqrt(NFIL/Real(fil_grid_dim_x))));
        const int fil_grid_dim_z = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x*fil_grid_dim_y))));
        const Real fil_grid_step_x = 90.0; //(0.1 + SCALED_BEAT_AMPLITUDE)*L; //2.0*L*sqrt(PI/5.6);
        const Real fil_grid_step_y = fil_grid_step_x;
        const Real fil_grid_step_z = fil_grid_step_x;

      #endif

    #elif DEFINED_BUT_EMPTY(FIL_LATTICE_X_NUM)

      // Only the y-size was provided.
      const int fil_grid_dim_y = std::min<int>(NFIL, FIL_LATTICE_Y_NUM);

      #if RECTANGULAR_SEEDING

        const int fil_grid_dim_x = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_y))));

      #elif HEXAGONAL_SEEDING

        const int fil_grid_dim_x = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_y-1))));

      #endif

    #else

      // Only the x-size was provided, or both sizes were provided. In case this latter option
      // doesn't account for all filaments, we ignore the provided FIL_LATTICE_Y_NUM and calculate the y-size for ourselves.
      #if !(FCM_LATTICE_SEEDING or FCM_RECTANGULAR_SEEDING)
        const int fil_grid_dim_x = std::min<int>(NFIL, FIL_LATTICE_X_NUM);
      #endif
      
      #if RECTANGULAR_SEEDING

        const int fil_grid_dim_y = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x))));

      #elif HEXAGONAL_SEEDING

        const int fil_grid_dim_y = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x-1))));
      #elif FCM_LATTICE_SEEDING or FCM_RECTANGULAR_SEEDING

        #if FCM_LATTICE_SEEDING
          const int fil_grid_dim_x = int(ceil(cbrt(Real(NFIL))));
          const int fil_grid_dim_y = std::max<int>(1, int(ceil(sqrt(NFIL/Real(fil_grid_dim_x)))));
          const int fil_grid_dim_z = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x*fil_grid_dim_y))));
        #elif FCM_RECTANGULAR_SEEDING
          const int fil_grid_dim_x = int(ceil(sqrt(Real(NFIL))))  ;
          const int fil_grid_dim_y = std::max<int>(1, int(ceil(NFIL/Real(fil_grid_dim_x))));
          const int fil_grid_dim_z = 1;

        #endif
        
        Real fil_grid_step_x;
        std::ifstream in("separation.dat"); // input
        in >> fil_grid_step_x;
        Real fil_grid_step_y = fil_grid_step_x;
        Real fil_grid_step_z = fil_grid_step_x;

      #elif TWOFIL_SEEDING
        // Place holder
        const int fil_grid_dim_y = 0.0;
      
      #endif

    #endif

    #if RECTANGULAR_SEEDING

      const Real fil_grid_step_y = FIL_LATTICE_Y_SPACING;

    #elif HEXAGONAL_SEEDING

      const Real fil_grid_step_y = 0.5*FIL_LATTICE_Y_SPACING;  // Gives us beat-wise separations of 2*fil_grid_step_y = FIL_LATTICE_Y_SPACING

    #endif

    #if !(FCM_LATTICE_SEEDING or FCM_RECTANGULAR_SEEDING)
      Real fil_grid_step_x;
    #endif

    if (!bool(FIL_LATTICE_X_SPACING)){

      // The 'run-time' variant of the macro magic.
      // We need this because FIL_LATTICE_X_SPACING could be a floating-point value, and so the preprocessor cannot deal with it.
      // However, the compiler should optimise away the branching and the actual executable will be the same.

      // Note: We will also enter here if FIL_LATTICE_X_SPACING = 0. Since this is just as useless as leaving it blank,
      // I don't think this needs to be documented in the config file.
      #if RECTANGULAR_SEEDING

        fil_grid_step_x = fil_grid_step_y;

      #elif HEXAGONAL_SEEDING

        fil_grid_step_x = 2.0*fil_grid_step_y/std::sqrt(3.0);

      #endif

    } else {
      #if !(FCM_LATTICE_SEEDING or FCM_RECTANGULAR_SEEDING)
        fil_grid_step_x = Real(FIL_LATTICE_X_SPACING); // Cast to Real lets it compile even if FIL_LATTICE_X_SPACING is blank.
      #endif
    }

    const Real im = 0.5*(fil_grid_dim_x - 1.0);
    const Real jm = 0.5*(fil_grid_dim_y - 1.0);

    const Real dir[3] = {0.0, 0.0, 1.0};
    const Real strain_twist[3] = {0.0, 0.0, 0.0};

    #if RECTANGULAR_SEEDING

      for (int i = 0; i < fil_grid_dim_x; i++){
        for (int j = 0; j < fil_grid_dim_y; j++){

          const int fil_id = j + i*fil_grid_dim_y;

          if (fil_id < NFIL){

            filament_references[3*fil_id] = (i-im)*fil_grid_step_x;
            filament_references[3*fil_id + 1] = (j-jm)*fil_grid_step_y;
            filament_references[3*fil_id + 2] = BASE_HEIGHT_ABOVE_SURFACE;

            Real *const fil_x_address = &x_segs_address[3*fil_id*NSEG];
            Real *const fil_f_address = &f_segs_address[6*fil_id*NSEG];

            #if READ_INITIAL_CONDITIONS_FROM_BACKUP

              filaments[fil_id].initial_setup(&filament_references[3*fil_id], dir, strain_twist, &data_from_file[fil_id*data_per_fil], fil_x_address, fil_f_address, fil_id, body.q);

            #else

              filaments[fil_id].initial_setup(&filament_references[3*fil_id], dir, strain_twist, data_from_file, fil_x_address, fil_f_address, fil_id, body.q);

            #endif

          }

        }
      }

    #elif HEXAGONAL_SEEDING

      int fil_id = 0;

      for (int j = 0; j < fil_grid_dim_y; j++){

        const int even_row = int(j != 2*(j/2)); // Because of indices starting at 0
        const int num_cols = fil_grid_dim_x - even_row;

        for (int i = 0; i < num_cols; i++){

          if (fil_id < NFIL){

            filament_references[3*fil_id] = (0.5*even_row + i - im)*fil_grid_step_x;
            filament_references[3*fil_id + 1] = (j-jm)*fil_grid_step_y;
            filament_references[3*fil_id + 2] = BASE_HEIGHT_ABOVE_SURFACE;

            Real *const fil_x_address = &x_segs_address[3*fil_id*NSEG];
            Real *const fil_f_address = &f_segs_address[6*fil_id*NSEG];

            #if READ_INITIAL_CONDITIONS_FROM_BACKUP

              filaments[fil_id].initial_setup(&filament_references[3*fil_id], dir, strain_twist, &data_from_file[fil_id*data_per_fil], fil_x_address, fil_f_address, fil_id, body.q);

            #else

              filaments[fil_id].initial_setup(&filament_references[3*fil_id], dir, strain_twist, data_from_file, fil_x_address, fil_f_address, fil_id, body.q);

            #endif

          }

          fil_id++;

        }

      }

    #elif FCM_LATTICE_SEEDING or FCM_RECTANGULAR_SEEDING

      for (int i = 0; i < fil_grid_dim_x; i++){
        for (int j = 0; j < fil_grid_dim_y; j++){
          for (int k = 0; k < fil_grid_dim_z; k++){

            const int fil_id = k + (j + i*fil_grid_dim_y)*fil_grid_dim_z;

            if (fil_id < NFIL){

              filament_references[3*fil_id] = i*fil_grid_step_x + 0.5*fil_grid_step_x;
              filament_references[3*fil_id + 1] = j*fil_grid_step_y + 0.5*fil_grid_step_y;
              filament_references[3*fil_id + 2] = k*fil_grid_step_z + 0.5*fil_grid_step_z;

              Real *const fil_x_address = &x_segs_address[3*fil_id*NSEG];
              Real *const fil_f_address = &f_segs_address[6*fil_id*NSEG];

              #if READ_INITIAL_CONDITIONS_FROM_BACKUP

                filaments[fil_id].initial_setup(&filament_references[3*fil_id], dir, strain_twist, &data_from_file[fil_id*data_per_fil], fil_x_address, fil_f_address, fil_id, body.q);

              #else

                filaments[fil_id].initial_setup(&filament_references[3*fil_id], dir, strain_twist, data_from_file, fil_x_address, fil_f_address, fil_id, body.q);

              #endif
            }
          }
        }
      }
    
    #elif TWOFIL_SEEDING
      
      std::cout << "Two-filament seeding is not implemented for infinite plane wall." << std::endl;
      filament_references[0] = 0.0;
      filament_references[1] = 0.0;
      filament_references[2] = 1.3;
      Real *const fil_x_address1 = &x_segs_address[3*0*NSEG];
      Real *const fil_f_address1 = &f_segs_address[6*0*NSEG];
      filaments[0].initial_setup(&filament_references[0], dir, strain_twist, data_from_file, fil_x_address1, fil_f_address1, 0, body.q);

      filament_references[3] = FIL_SPACING*std::cos(TWOFIL_ANGLE);
      filament_references[4] = FIL_SPACING*std::sin(TWOFIL_ANGLE);
      filament_references[5] = 1.3;
      Real *const fil_x_address2 = &x_segs_address[3*1*NSEG];
      Real *const fil_f_address2 = &f_segs_address[6*1*NSEG];
      filaments[1].initial_setup(&filament_references[3], dir, strain_twist, data_from_file, fil_x_address2, fil_f_address2, 1, body.q);

    #endif



  #elif SADDLE_BODIES

    // NOT YET IMPLEMENTED.

  #elif SURFACE_OF_REVOLUTION_BODIES or ROD or RIGIDWALL

    std::string file_name_trunk = GENERATRIX_FILE_NAME+std::to_string(NFIL);

    #if EQUATORIAL_SEEDING

      file_name_trunk += "_equatorial";

    #elif PLATY_SEEDING

      file_name_trunk += "_platy";

    #endif

    std::ifstream pos_file(file_name_trunk + ".seed");
    std::ifstream polar_file(file_name_trunk + ".polar_dir");
    std::ifstream azi_file(file_name_trunk + ".azi_dir");
    std::ifstream normal_file(file_name_trunk + ".normal");

    if (pos_file.good() && polar_file.good() && azi_file.good() && normal_file.good()){
      
      std::cout << std::endl << std::endl << "Using " << file_name_trunk + ".seed" << " as fil input file." << std::endl;
        
      for (int i = 0; i < 3*NFIL; i++){

        pos_file >> filament_references[i];
        polar_file >> polar_dir_refs[i];
        azi_file >> azi_dir_refs[i];
        normal_file >> normal_refs[i];

      }


    } else {

      seed_filaments(&filament_references[0], &polar_dir_refs[0], &azi_dir_refs[0], &normal_refs[0]);

    }

    // DEBUGING BEGIN

      // check_seeding(&filament_references[0], &polar_dir_refs[0], &azi_dir_refs[0], &normal_refs[0]);    

    // DEBUGING END
    

    // Force to overwrite the seeding file
    #if RIGIDWALL or ICOSA_SEEDING
      seed_filaments(&filament_references[0], &polar_dir_refs[0], &azi_dir_refs[0], &normal_refs[0]);
    #endif


    #if SURFACE_OF_REVOLUTION_BODIES

      // The seeding functions work on unit-length bodies, so the scaling must be done after we read or calculate.
      for (int i = 0; i < NFIL; i++){

        filament_references[3*i] *= AXIS_DIR_BODY_LENGTH;
        filament_references[3*i + 1] *= AXIS_DIR_BODY_LENGTH;
        filament_references[3*i + 2] *= AXIS_DIR_BODY_LENGTH;

        filament_references[3*i] += (RBLOB + BASE_HEIGHT_ABOVE_SURFACE)*normal_refs[3*i];
        filament_references[3*i + 1] += (RBLOB + BASE_HEIGHT_ABOVE_SURFACE)*normal_refs[3*i + 1];
        filament_references[3*i + 2] += (RBLOB + BASE_HEIGHT_ABOVE_SURFACE)*normal_refs[3*i + 2];

      }
    #endif

    pos_file.close();
    polar_file.close();
    azi_file.close();
    normal_file.close();

    const Real strain_twist[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < NFIL; i++) {

      Real *const fil_x_address = &x_segs_address[3*i*NSEG];
      Real *const fil_f_address = &f_segs_address[6*i*NSEG];
      
      #if RIGIDWALL and HEXAGONAL_WALL_SEEDING
        const Real dir[3] = {0.0, 0.0, 1.0};
      #else
        const Real *const dir = &normal_refs[3*i];
      #endif
      // force to use normal from seeding
      // const Real *const dir = &normal_refs[3*i];

      // dir assumes the body.q is in the default position upright.
      // if body.q is not pointing the default position, say, read from a file, then we need to rotate dir accordingly,
      // eventually the rotated dir, called dir_in, is passed to initialise filaments.
      // deprecated, instead body.q is passed to initial filaments (20241001)
      // matrix dir_matrix(3,1);
      // dir_matrix(0,0) = dir[0];
      // dir_matrix(1,0) = dir[1];
      // dir_matrix(2,0) = dir[2];
      // dir_matrix = body.q.rot_mat()*dir_matrix;
      // Real dir_in[3] = {dir_matrix(0,0), dir_matrix(1,0), dir_matrix(2,0)};

      const Real pos[3] = {body.x[0] + filament_references[3*i], body.x[1] + filament_references[3*i + 1], body.x[2] + filament_references[3*i + 2]};
      
      #if READ_INITIAL_CONDITIONS_FROM_BACKUP

        filaments[i].initial_setup(pos, dir, strain_twist, &data_from_file[i*data_per_fil], fil_x_address, fil_f_address, i, body.q);

      #else

        filaments[i].initial_setup(pos, dir, strain_twist, data_from_file, fil_x_address, fil_f_address, i, body.q);

      #endif

    }

  #elif TORUS_BODIES

    // NOT YET IMPLEMENTED.

  #endif

}

void swimmer::initial_guess(const int nt){

  #if (PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

    // These types of simulation are explicit, so we want to the body to stay where it is rather than guess an updated state.
    // It does need this one small bit of index-updating, however.
    body.qm1 = body.q;

  #else

    body.initial_guess(nt);

  #endif

  matrix R = body.q.rot_mat();

  for (int n = 0; n < NFIL; n++){

    const matrix base_pos = matrix(3, 1, body.x) + R*matrix(3, 1, &filament_references[3*n]);

    // The filaments won't actually make guesses if they're representing prescribed-motion cilia.
    // This call will just give them the state of the body, as well as let them change their
    // frequency etc. if we're including any phototaxis effects.
    filaments[n].initial_guess(nt, &base_pos.data[0], body.u);

  }

  #if PRESCRIBED_CILIA

    // This has to be called AFTER the filaments have made their initial guess.
    make_precon_mat();

  #endif

}

void swimmer::forces_and_torques(const int nt, int id){

  #if !PRESCRIBED_CILIA

    f.zero(); // Force and torque on the swimmer body.

    for (int n = 0; n < NFIL; n++){

      filaments[n].internal_forces_and_torques(nt);

      #if !PRESCRIBED_BODY_VELOCITIES

        // Account for the free-swimming condition by SUBTRACTING the total force and torque
        // due to the filament segments.
        for (int m = 0; m < NSEG; m++){

          const Real *const f_seg = &filaments[n].f[6*m];

          f(0) -= f_seg[0];
          f(1) -= f_seg[1];
          f(2) -= f_seg[2];

          const Real xdiff = filaments[n].segments[m].x[0] - body.x[0];
          const Real ydiff = filaments[n].segments[m].x[1] - body.x[1];
          const Real zdiff = filaments[n].segments[m].x[2] - body.x[2];

          f(3) -= f_seg[3] + ydiff*f_seg[2] - zdiff*f_seg[1];
          f(4) -= f_seg[4] + zdiff*f_seg[0] - xdiff*f_seg[2];
          f(5) -= f_seg[5] + xdiff*f_seg[1] - ydiff*f_seg[0];

        }

      #endif

      // Now apply any EXTERNAL forces and torques -- i.e. those not considered part of the
      // swimmer's internal activity -- to the filament. This must occur after the above
      // calculation because these external forces do not contribute to the free-swimming
      // condition.

    }

    #if !PRESCRIBED_BODY_VELOCITIES

      // const matrix R = q.rot_mat();
      // const matrix ref(3, 1, &blob_references[3*i]);
      // const matrix ext_f = R*ref;

      // x_array[3*i] = pos(0);
      // x_array[3*i + 1] = pos(1);
      // x_array[3*i + 2] = pos(2);

      #if ROD
        f(0) += Real(NBLOB);
      #endif
      

      // Finally, add any external forces on the blobs, and the induced torques on body, to f.

      f(2) += 1000;


    #endif

  #endif

}

#if !(PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

  void swimmer::prepare_jacobian_inv(const int nt){
    
    // printf("prepare_jacobian_inv pass1\n");
    
    for (int i = 0; i < NFIL; i++){

      filaments[i].invert_approx_jacobian(nt);

    }

    // printf("prepare_jacobian_inv pass2\n");

    #if !INFINITE_PLANE_WALL

      // Use aliases so we don't have to write body.(...) everywhere.
      const matrix& Q_init = body.Q_init;
      const Real *const u = body.u;
      const Real *const x = body.x;

      matrix& schur_mat = schur_mat_inv; // Use an alias then invert in-place.

      // printf("prepare_jacobian_inv matrix pass1\n");

      #if USE_BROYDEN_FOR_EVERYTHING

        #if PRESCRIBED_BODY_VELOCITIES

          // The body state equations only depend on the body state variables -- the velocities
          // are prescribed and so not even the blob-force Lagrange multipliers contribute.
          // Thus the Schur complement matrix will just be the identity, and hence so will its inverse.
          // As we only multiply by the inverse, we can just ignore it and leave this function.
          return;

        #else

          schur_mat.zero();

          const Real blob_mob_fac = (nt < NUM_EULER_STEPS) ? -6.0*PI*MU*RBLOB/DT : -9.0*PI*MU*RBLOB/DT; // We can absorb the DT factor here because they never appear separately.

          schur_mat(0, 0) = -NBLOB*blob_mob_fac;
          schur_mat(1, 1) = -NBLOB*blob_mob_fac;
          schur_mat(2, 2) = -NBLOB*blob_mob_fac;

          // printf("prepare_jacobian_inv matrix pass2\n");

          for (int m = 0; m < NBLOB; m++){

            const Real blob_force[3] = {body.blob_forces[3*m], body.blob_forces[3*m + 1], body.blob_forces[3*m + 2]};
            const matrix blob_diff = Q_init*matrix(3, 1, &body.blob_references[3*m]);
            const matrix rcross_blob_diff = rcross(blob_diff);

            schur_mat.add_to_block(3, 3, 3, 3, rcross(blob_force)*rcross_blob_diff); // Original bottom-right terms in Jacobian. No blob_mob_fac here.

            // Even better approx. would be blob_mob_fac*(I - rcross(Omega_as_a_function_of_u))*rcross_blob_diff, but this would require (a tiny amount of) extra storage to replicate during jacobian_inv_mult(...)
            schur_mat.subtract_from_block(0, 3, 3, 3, blob_mob_fac*rcross_blob_diff);

            schur_mat.add_to_block(3, 0, 3, 3, blob_mob_fac*rcross_blob_diff); // This is correct independently of the approx. used for the schur_mat(0, 3, size(3,3)) block.

            schur_mat.add_to_block(3, 3, 3, 3, blob_mob_fac*rcross_blob_diff*rcross_blob_diff); // Right-most matrix should match the approx. used for the schur_mat(0, 3, size(3,3)) block.

          }

          // printf("prepare_jacobian_inv pass3\n");

          for (int n = 0; n < NFIL; n++){

            const matrix fil_base_disp = Q_init*matrix(3, 1, &filament_references[3*n]);

            schur_mat.add_to_block(3, 3, 3, 3, rcross(filaments[n].tether_lambda)*rcross(fil_base_disp));

            const Real u_cross_lambda[3] = {u[1]*filaments[n].clamp_lambda[2] - u[2]*filaments[n].clamp_lambda[1],
                                              u[2]*filaments[n].clamp_lambda[0] - u[0]*filaments[n].clamp_lambda[2],
                                              u[0]*filaments[n].clamp_lambda[1] - u[1]*filaments[n].clamp_lambda[0]};

            schur_mat.add_to_block(3, 3, 3, 3, 0.5*rcross(filaments[n].clamp_lambda) + (rcross(u_cross_lambda) - rcross(u)*rcross(filaments[n].clamp_lambda))/12.0);

            #if INSTABILITY_CILIA

              matrix t_base(3,1), t_end(3,1);
              filaments[n].segments[0].tangent(t_base);
              filaments[n].segments[NSEG-1].tangent(t_end);

              schur_mat.add_to_block(3, 3, 3, 3, rcross(-END_FORCE_MAGNITUDE*t_end)*rcross(0.5*DL*t_base));

            #endif

            matrix B(6*NSEG, 6);
            B.zero();
            matrix D(6, 6*NSEG);
            D.zero();

            D(0,0) = 1.0;
            D(1,1) = 1.0;
            D(2,2) = 1.0;

            D.set_block(3, 0, 3, 3, rcross(-fil_base_disp));

            D.set_block(3, 3*NSEG, 3, 3, rcross(u)*rcross(u)/12.0 - 0.5*rcross(u));
            D(3, 3*NSEG) += 1.0;
            D(4, 3*NSEG + 1) += 1.0;
            D(5, 3*NSEG + 2) += 1.0;

            #if INSTABILITY_CILIA

              D.set_block(0, 6*NSEG - 3, 3, 3, rcross(-END_FORCE_MAGNITUDE*t_end));

              matrix diff(3,1);
              diff(0) = filaments[n].segments[NSEG-1].x[0] - x[0];
              diff(1) = filaments[n].segments[NSEG-1].x[1] - x[1];
              diff(2) = filaments[n].segments[NSEG-1].x[2] - x[2];
              D.set_block(3, 6*NSEG - 3, 3, 3, rcross(0.5*DL*t_end - diff)*D.get_block(0, 6*NSEG - 3, 3, 3));

              for (int m = 1; m < NSEG-1; m++){

                matrix t(3,1);
                filaments[n].segments[m].tangent(t);
                D.set_block(3, 3*(NSEG + m), 3, 3, D.get_block(0, 6*NSEG - 3, 3, 3)*rcross(DL*t));

              }

            #endif

            jacobian_B_blocks[n] = D*filaments[n].inverse_jacobian;

            B.set_block(3*NSEG, 3, 3, 3, filaments[n].elastic_clamping_block1);
            B.set_block(3*NSEG + 3, 3, 3, 3, filaments[n].elastic_clamping_block2);

            const Real disp_norm = sqrt(fil_base_disp(0)*fil_base_disp(0) + fil_base_disp(1)*fil_base_disp(1) + fil_base_disp(2)*fil_base_disp(2));

            const matrix Du = rcross((0.5*DL + disp_norm)*fil_base_disp/disp_norm);

            for (int m = 0; m < NSEG; m++){

              B(3*m, 0) = 1.0;
              B(3*m + 1, 1) = 1.0;
              B(3*m + 2, 2) = 1.0;

              B.set_block(3*m, 3, 3, 3, Du);

            }

            schur_mat -= jacobian_B_blocks[n]*B;

          }

          // printf("prepare_jacobian_inv pass4\n");

        #endif

      #else

        const Real dt_fac = (nt < NUM_EULER_STEPS) ? -DT : -2.0*DT/3.0;
        const matrix body_lie_cross = -rcross(u);

        // This approx. doesn't require knowledge of the mobility solver.
        // It also means that the term rcross(cross_body_omega*u)/12.0 in the following block addition vanishes.
        const matrix cross_body_omega = -body_lie_cross/dt_fac;

        schur_mat.identity();
        schur_mat.add_to_block(3, 3, 3, 3, dt_fac*(body_lie_cross*cross_body_omega/12.0 - 0.5*cross_body_omega));

        #if !PRESCRIBED_BODY_VELOCITIES

          matrix t(3,1);

          matrix omega_lie_deriv(3,3);
          omega_lie_deriv.zero();

          for (int i = 0; i < NFIL; i++){

            const matrix fil_base_disp = Q_init*matrix(3, 1, &filament_references[3*i]);

            omega_lie_deriv -= rcross(filaments[i].tether_lambda)*rcross(fil_base_disp);

            const Real u_cross_lambda[3] = {u[1]*filaments[i].clamp_lambda[2] - u[2]*filaments[i].clamp_lambda[1],
                                              u[2]*filaments[i].clamp_lambda[0] - u[0]*filaments[i].clamp_lambda[2],
                                              u[0]*filaments[i].clamp_lambda[1] - u[1]*filaments[i].clamp_lambda[0]};

            omega_lie_deriv += 0.5*rcross(filaments[i].clamp_lambda) - (rcross(u_cross_lambda) + body_lie_cross*rcross(filaments[i].clamp_lambda))/12.0;

            #if INSTABILITY_CILIA

              filaments[i].segments[NSEG-1].tangent(t);

              const Real disp_norm = sqrt(fil_base_disp(0)*fil_base_disp(0) + fil_base_disp(1)*fil_base_disp(1) + fil_base_disp(2)*fil_base_disp(2));

              omega_lie_deriv += rcross(END_FORCE_MAGNITUDE*t)*rcross((0.5*DL + disp_norm)*fil_base_disp/disp_norm);

            #endif

          }

          matrix body_mobility = body_mobility_reference;
          const matrix Q_init_inv = transpose(Q_init);
          body_mobility.set_block(0, 0, 3, 3, Q_init*body_mobility.get_block(0, 0, 3, 3)*Q_init_inv);
          body_mobility.set_block(0, 3, 3, 3, Q_init*body_mobility.get_block(0, 3, 3, 3)*Q_init_inv);
          body_mobility.set_block(3, 0, 3, 3, Q_init*body_mobility.get_block(3, 0, 3, 3)*Q_init_inv);
          body_mobility.set_block(3, 3, 3, 3, Q_init*body_mobility.get_block(3, 3, 3, 3)*Q_init_inv);

          omega_lie_deriv = body_mobility.get_block(3, 3, 3, 3)*omega_lie_deriv;

          // Add contribution to omega_lie_deriv from segments here <---

          schur_mat.add_to_block(3, 3, 3, 3, dt_fac*(omega_lie_deriv - 0.5*body_lie_cross*omega_lie_deriv + body_lie_cross*body_lie_cross*omega_lie_deriv/12.0));

          // This concludes the self-dependence of the rigid body Lie algebra update equations
          // and we now focus on the derivatives w.r.t. filament variables.
          matrix D_inv(3,3), D_inv_transpose(3,3);
          D_inv.identity();
          D_inv_transpose.identity();
          D_inv += body_lie_cross*body_lie_cross/12.0 - 0.5*body_lie_cross;
          D_inv_transpose += body_lie_cross*body_lie_cross/12.0 + 0.5*body_lie_cross;

          matrix A(6*NSEG, 6);
          A.zero();

          for (int n = 0; n < NSEG; n++){

            A(3*n, 0) = 1.0;
            A(3*n + 1, 1) = 1.0;
            A(3*n + 2, 2) = 1.0;

          }

          for (int i = 0; i < NFIL; i++){

            matrix B(6, 6*NSEG);
            B.zero();

            B(0, 0) = -1.0;
            B(1, 1) = -1.0;
            B(2, 2) = -1.0;

            const matrix fil_base_disp = Q_init*matrix(3, 1, &filament_references[3*i]);
            B.set_block(3, 0, 3, 3, rcross(fil_base_disp));

            B.set_block(3, 3*NSEG, 3, 3, -D_inv_transpose);

            const std::vector<segment>& segments = filaments[i].segments;

            #if INSTABILITY_CILIA

              segments[NSEG-1].tangent(t);
              B.set_block(0, 6*NSEG - 3, 3, 3, rcross(END_FORCE_MAGNITUDE*t));

              matrix diff(3,1);
              diff(0) = segments[NSEG-1].x[0] - x[0];
              diff(1) = segments[NSEG-1].x[1] - x[1];
              diff(2) = segments[NSEG-1].x[2] - x[2];
              B.set_block(3, 6*NSEG - 3, 3, 3, rcross(0.5*DL*t - diff)*B.get_block(0, 6*NSEG - 3, 3, 3));

              for (int m = 1; m < NSEG-1; m++){

                matrix t(3,1);
                segments[m].tangent(t);
                B.set_block(3, 3*(NSEG + m), 3, 3, B.get_block(0, 6*NSEG - 3, 3, 3)*rcross(DL*t));

              }

            #endif

            B = body_mobility*B;

            #if SURFACE_OF_REVOLUTION_BODIES or ROD

              if(hasEnding(std::string(GENERATRIX_FILE_NAME), std::string("sphere"))){
              // if (std::string(GENERATRIX_FILE_NAME) == std::string("sphere")){

                // If the body is a sphere, we can make corrections to dependencies on the segment variables
                // using the RPY expression for interacting spheres of different sizes.
                const Real R = 0.5*AXIS_DIR_BODY_LENGTH;

                for (int n = 0; n < NSEG; n++){

                  matrix Mtt(3,3), Mtr(3,3), Mrr(3,3), I(3,3), rr(3,3);

                  I.identity();

                  Real rhat[3] = {segments[n].x[0] - x[0], segments[n].x[1] - x[1], segments[n].x[2] - x[2]};
                  const Real r = sqrt(rhat[0]*rhat[0] + rhat[1]*rhat[1] + rhat[2]*rhat[2]);
                  rhat[0] /= r;
                  rhat[1] /= r;
                  rhat[2] /= r;
                  rr(0,0) = rhat[0]*rhat[0];
                  rr(1,1) = rhat[1]*rhat[1];
                  rr(2,2) = rhat[2]*rhat[2];
                  rr(0,1) = rhat[0]*rhat[1];
                  rr(1,0) = rr(0,1);
                  rr(0,2) = rhat[0]*rhat[2];
                  rr(2,0) = rr(0,2);
                  rr(1,2) = rhat[2]*rhat[1];
                  rr(2,1) = rr(1,2);


                  Mtt = ((1.0 + (R*R + RSEG*RSEG)/(3.0*r*r))*I + (1.0 - (R*R + RSEG*RSEG)/(r*r))*rr)/(8.0*PI*MU*r);
                  Mtr = rcross(rhat)/(8.0*PI*MU*r*r);
                  Mrr = (3.0*rr - I)/(16.0*PI*MU*r*r*r);

                  // TO-DO: Add dependence on general Lie algebra elements?

                  segments[n].tangent(t);
                  const matrix temp = rcross(t);

                  if (n == 0){

                    B.add_to_block(0, 0, 3, 3, Mtt); // tether_lambda
                    B.add_to_block(3, 0, 3, 3, Mtr);

                    B.add_to_block(0, 3*NSEG, 3, 3, Mtr*D_inv_transpose); // clamp_lambda
                    B.add_to_block(3, 3*NSEG, 3, 3, Mrr*D_inv_transpose);

                    B.add_to_block(0, 3, 3, 3, 0.5*DL*Mtr*temp - Mtt); // lambda_inex
                    B.add_to_block(3, 3, 3, 3, 0.5*DL*Mrr*temp - Mtr);

                  } else if (n == NSEG-1){

                    #if INSTABILITY_CILIA

                      B.add_to_block(0, 6*NSEG-3, 3, 3, -END_FORCE_MAGNITUDE*Mtt*temp); // distal Lie algebra element
                      B.add_to_block(3, 6*NSEG-3, 3, 3, -END_FORCE_MAGNITUDE*Mtr*temp);

                    #endif

                    B.add_to_block(0, 3*NSEG-3, 3, 3, 0.5*DL*Mtr*temp + Mtt); // lambda_inex
                    B.add_to_block(3, 3*NSEG-3, 3, 3, 0.5*DL*Mrr*temp + Mtr);

                  } else {

                    B.add_to_block(0, 3*n, 3, 3, 0.5*DL*Mtr*temp + Mtt);
                    B.add_to_block(3, 3*n, 3, 3, 0.5*DL*Mrr*temp + Mtr);

                    B.add_to_block(0, 3*(n+1), 3, 3, 0.5*DL*Mtr*temp - Mtt);
                    B.add_to_block(3, 3*(n+1), 3, 3, 0.5*DL*Mrr*temp - Mtr);

                  }

                }

              }

            #endif // TO-DO: Add the non-spherical approximation

            for (int n = 0; n < NSEG; n++){

              B.set_block(3, 3*n, 3, 3, D_inv*B.get_block(3, 3*n, 3, 3));
              B.set_block(3, 3*(n + NSEG), 3, 3, D_inv*B.get_block(3, 3*(n + NSEG), 3, 3));

            }

            B *= dt_fac;
            jacobian_B_blocks[i] = B*filaments[i].inverse_jacobian;

            A.set_block(3*NSEG, 3, 3, 3, filaments[i].elastic_clamping_block1);
            A.set_block(3*NSEG + 3, 3, 3, 3, filaments[i].elastic_clamping_block2);

            const Real disp_norm = sqrt(fil_base_disp(0)*fil_base_disp(0) + fil_base_disp(1)*fil_base_disp(1) + fil_base_disp(2)*fil_base_disp(2));
            const matrix D = rcross((0.5*DL + disp_norm)*fil_base_disp/disp_norm);

            for (int n = 0; n < NSEG; n++){

              A.set_block(3*n, 3, 3, 3, D);

            }

            schur_mat -= jacobian_B_blocks[i]*A;

          } // End of loop over filaments

        #endif

      #endif

      schur_mat.invert(); // Stored in schur_mat_inv via aliasing.

      // printf("prepare_jacobian_inv pass5\n");

    #endif

  }

  matrix swimmer::jacobian_inv_mult(const matrix& in, const int nt) const {

    matrix out(in);

    const matrix& Q_init = body.Q_init; // alias

    #if INFINITE_PLANE_WALL

      for (int n = 0; n < NFIL; n++){

        out.set_block(6*NSEG*n, 6*NSEG, filaments[n].inverse_jacobian*in.get_block(6*NSEG*n, 6*NSEG));

      }

    #else

      #if USE_BROYDEN_FOR_EVERYTHING

        matrix b1 = in.get_block(0, 6*NFIL*NSEG);

        const Real blob_mob_fac = -6.0*PI*MU*RBLOB;

        #if PRESCRIBED_BODY_VELOCITIES

          // The (6*NFIL*NSEG)-to-(6*NFIL*NSEG + 5) block is already correct.

          out.multiply_block(6*NFIL*NSEG + 6, 3*NBLOB, blob_mob_fac);

        #else

          matrix b2 = in.get_block(6*NFIL*NSEG + 6, 3*NBLOB);
          matrix b3 = in.get_block(6*NFIL*NSEG, 6);

          for (int n = 0; n < NFIL; n++){

            b3 -= jacobian_B_blocks[n]*b1.get_block(6*n*NSEG, 6*NSEG);

          }

          for (int m = 0; m < NBLOB; m++){

            b3.subtract_from_block(0, 3, blob_mob_fac*b2.get_block(3*m, 3));

            const matrix blob_disp = Q_init*matrix(3, 1, &body.blob_references[3*m]);
            b3(3) -= blob_mob_fac*(blob_disp(1)*b2(3*m + 2) - blob_disp(2)*b2(3*m + 1));
            b3(4) -= blob_mob_fac*(blob_disp(2)*b2(3*m) - blob_disp(0)*b2(3*m + 2));
            b3(5) -= blob_mob_fac*(blob_disp(0)*b2(3*m + 1) - blob_disp(1)*b2(3*m));

          }

          out.set_block(6*NFIL*NSEG, 6, schur_mat_inv*b3);

          const Real dt_fac = (nt < NUM_EULER_STEPS) ? 1.0/DT : 1.5/DT;

          for (int m = 0; m < NBLOB; m++){

            const matrix blob_disp = Q_init*matrix(3, 1, &body.blob_references[3*m]);
            matrix out_cross_disp(3,1);
            out_cross_disp(0) = out(6*NFIL*NSEG + 4)*blob_disp(2) - out(6*NFIL*NSEG + 5)*blob_disp(1);
            out_cross_disp(1) = out(6*NFIL*NSEG + 5)*blob_disp(0) - out(6*NFIL*NSEG + 3)*blob_disp(2);
            out_cross_disp(2) = out(6*NFIL*NSEG + 3)*blob_disp(1) - out(6*NFIL*NSEG + 4)*blob_disp(0);

            // Ideally, the cross product bit at the end should match the approx. used for the block schur_mat(0, 3, size(3,3)) when preparing the Jacobian.
            b2.subtract_from_block(3*m, 3, dt_fac*(out.get_block(6*NFIL*NSEG, 3) + out_cross_disp));

          }

          out.set_block(6*NFIL*NSEG + 6, 3*NBLOB, blob_mob_fac*b2);

        #endif

        matrix B(6*NSEG, 6);
        B.zero();

        for (int m = 0; m < NSEG; m++){

          B(3*m, 0) = 1.0;
          B(3*m + 1, 1) = 1.0;
          B(3*m + 2, 2) = 1.0;

        }

        for (int n = 0; n < NFIL; n++){

          B.set_block(3*NSEG, 3, 3, 3, filaments[n].elastic_clamping_block1);
          B.set_block(3*NSEG + 3, 3, 3, 3, filaments[n].elastic_clamping_block2);

          const matrix fil_base_disp = Q_init*matrix(3, 1, &filament_references[3*n]);
          const Real disp_norm = sqrt(fil_base_disp(0)*fil_base_disp(0) + fil_base_disp(1)*fil_base_disp(1) + fil_base_disp(2)*fil_base_disp(2));
          const matrix D = rcross((0.5*DL + disp_norm)*fil_base_disp/disp_norm);

          for (int m = 0; m < NSEG; m++){

            B.set_block(3*m, 3, 3, 3, D);

          }

          b1.subtract_from_block(6*NSEG*n, 6*NSEG, B*out.get_block(6*NFIL*NSEG, 6));

          out.set_block(6*NSEG*n, 6*NSEG, filaments[n].inverse_jacobian*b1.get_block(6*NSEG*n, 6*NSEG));

        }

      #else

        matrix b2 = in.get_block(6*NFIL*NSEG, 6);

        #if !PRESCRIBED_BODY_VELOCITIES

          for (int i = 0; i < NFIL; i++){

            b2 -= jacobian_B_blocks[i]*in.get_block(6*NSEG*i, 6*NSEG);

          }

        #endif

        out.set_block(6*NFIL*NSEG, 6, schur_mat_inv*b2);

        matrix A(6*NSEG, 6);
        A.zero();

        for (int n = 0; n < NSEG; n++){

          A(3*n, 0) = 1.0;
          A(3*n + 1, 1) = 1.0;
          A(3*n + 2, 2) = 1.0;

        }

        for (int i = 0; i < NFIL; i++){

          A.set_block(3*NSEG, 3, 3, 3, filaments[i].elastic_clamping_block1);
          A.set_block(3*NSEG + 3, 3, 3, 3, filaments[i].elastic_clamping_block2);

          const matrix fil_base_disp = Q_init*matrix(3, 1, &filament_references[3*i]);
          const Real disp_norm = sqrt(fil_base_disp(0)*fil_base_disp(0) + fil_base_disp(1)*fil_base_disp(1) + fil_base_disp(2)*fil_base_disp(2));
          const matrix D = rcross((0.5*DL + disp_norm)*fil_base_disp/disp_norm);

          for (int n = 0; n < NSEG; n++){

            A.set_block(3*n, 3, 3, 3, D);

          }

          out.set_block(6*NSEG*i, 6*NSEG, filaments[i].inverse_jacobian*(in.get_block(6*NSEG*i, 6*NSEG) - A*out.get_block(6*NFIL*NSEG, 6)));

        }

      #endif

    #endif

    return out;

  }

  void swimmer::update(const Real *const swimmer_update){

    #if !INFINITE_PLANE_WALL

      body.update(&swimmer_update[6*NFIL*NSEG]);

      const matrix R = body.q.rot_mat();

    #endif

    for (int i = 0; i < NFIL; i++){

      #if !INFINITE_PLANE_WALL

        const matrix base_pos = matrix(3, 1, body.x) + R*matrix(3, 1, &filament_references[3*i]);

        filaments[i].accept_state_from_rigid_body(&base_pos.data[0], body.u);

      #endif

      filaments[i].update(&swimmer_update[6*i*NSEG]);

    }

  }

#endif

void swimmer::end_of_step(const int nt){

  for (int i = 0; i < NFIL; i++){

    filaments[i].end_of_step(nt);

  }

}

void swimmer::write_reference_positions() const {

  body.write_reference_positions();

  std::ofstream fil_ref_file(SIMULATION_NAME+std::string("_fil_references.dat"));

  for (int n = 0; n < 3*NFIL; n++){

    fil_ref_file << filament_references[n] << " " ;

  }

  fil_ref_file << "\n";
  fil_ref_file.close();

  #if PRESCRIBED_CILIA

    std::ofstream fil_q_file(SIMULATION_NAME+std::string("_fil_q.dat"));
    for (int n = 0; n < NFIL; n++){

      fil_q_file << filaments[n].body_q_ref.scalar_part << " " << filaments[n].body_q_ref.vector_part[0] << " " << filaments[n].body_q_ref.vector_part[1] << " " << filaments[n].body_q_ref.vector_part[2] << " " ;

    }
    fil_q_file << "\n";
    fil_q_file.close();

  #endif

}

void swimmer::write_data(std::ofstream& seg_state_file, std::ofstream& body_state_file) const {

  body.write_data(body_state_file);

  #if OUTPUT_FORCES

    for (int n = 0; n < NFIL; n++){

      filaments[n].write_data(seg_state_file);

    }

  #endif

}

void swimmer::write_backup(std::ofstream& backup_file) const {

  for (int n = 0; n < NFIL; n++){

    filaments[n].write_backup(backup_file);

  }

  // Do this after the filaments backup so we can start GMRES from Broyden-only backups.
  body.write_backup(backup_file);

}

#if PRESCRIBED_CILIA

  void swimmer::make_precon_mat(){

    #if !PRESCRIBED_BODY_VELOCITIES

      KTMinvK_inv.zero();

      const Real seg_mob_fac = 6.0*PI*MU*RSEG;
      const Real blob_mob_fac = 6.0*PI*MU*RBLOB;

      const Real eye_fac = Real(NFIL*NSEG)*seg_mob_fac + Real(NBLOB)*blob_mob_fac;
      KTMinvK_inv(0, 0) = eye_fac;
      KTMinvK_inv(1, 1) = eye_fac;
      KTMinvK_inv(2, 2) = eye_fac;

      const Real *const x = body.x;

      for (int n = 0; n < NFIL; n++){

        const std::vector<segment>& segments = filaments[n].segments;

        for (int m = 0; m < NSEG; m++){

          const Real diff[3] = {segments[m].x[0] - x[0], segments[m].x[1] - x[1], segments[m].x[2] - x[2]};
          const matrix rcross_mat = rcross(diff);

          KTMinvK_inv.add_to_block(0, 3, 3, 3, seg_mob_fac*rcross_mat);
          KTMinvK_inv.subtract_from_block(3, 3, 3, 3, seg_mob_fac*rcross_mat*rcross_mat); // minus sign for transpose

        }

      }

      const matrix Q = body.q.rot_mat();

      for (int n = 0; n < NBLOB; n++){

        const matrix rcross_mat = rcross(Q*matrix(3, 1, &body.blob_references[3*n]));

        KTMinvK_inv.add_to_block(0, 3, 3, 3, blob_mob_fac*rcross_mat);
        KTMinvK_inv.subtract_from_block(3, 3, 3, 3, blob_mob_fac*rcross_mat*rcross_mat); // minus sign for transpose

      }

      KTMinvK_inv.set_block(3, 0, 3, 3, -KTMinvK_inv.get_block(0, 3, 3, 3)); // minus sign for transpose

      #if DYNAMIC_PHASE_EVOLUTION and !DYNAMIC_SHAPE_ROTATION

        // In this case, we're not actually storing KTMinvK_inv anymore, but rather
        // the 6x6 matrix that appears in the Schur complement expression for it.
        matrix C_Ainv_B(6,6);
        C_Ainv_B.zero();

        // Each filament/cilium contributes independently
        for (int n = 0; n < NFIL; n++){

          Real Knormsq = 0.0;
          matrix v1(3,1), v2(3,1);
          v1.zero();
          v2.zero();

          for (int m = 0; m < NSEG; m++){

            matrix k_phi(3,1);
            k_phi(0) = filaments[n].vel_dir_phase[3*m];
            k_phi(1) = filaments[n].vel_dir_phase[3*m + 1];
            k_phi(2) = filaments[n].vel_dir_phase[3*m + 2];

            v1 += k_phi;

            matrix diff(3,1);
            diff(0) = filaments[n].segments[m].x[0] - x[0];
            diff(1) = filaments[n].segments[m].x[1] - x[1];
            diff(2) = filaments[n].segments[m].x[2] - x[2];

            v2 += cross(diff, k_phi);

            Knormsq += dot(k_phi, k_phi);

          }

          C_Ainv_B.add_to_block(0, 0, 3, 3, v1*transpose(v1)/Knormsq);
          C_Ainv_B.add_to_block(3, 0, 3, 3, v2*transpose(v1)/Knormsq);
          C_Ainv_B.add_to_block(3, 3, 3, 3, v2*transpose(v2)/Knormsq);

        }

        // Use symmetry to finish off the new terms
        C_Ainv_B.set_block(0, 3, 3, 3, transpose(C_Ainv_B.get_block(3, 0, 3, 3)));

        // Finally, add on the new terms
        KTMinvK_inv -= seg_mob_fac*C_Ainv_B;

      #endif

      #if DYNAMIC_PHASE_EVOLUTION and DYNAMIC_SHAPE_ROTATION

        // In this case, we're not actually storing KTMinvK_inv anymore, but rather
        // the 6x6 matrix that appears in the Schur complement expression for it.
        matrix C_Ainv_B(6,6);
        C_Ainv_B.zero();

        // Each filament/cilium contributes independently
        for (int n = 0; n < NFIL; n++){

          Real c11 = 0.0, c22 = 0.0, c12 = 0.0;
          Real cinv_11 = 0.0, cinv_22 = 0.0, cinv_12 = 0.0;
          matrix v_phi(3,1), u_phi(3,1), v_theta(3,1), u_theta(3,1);
          v_phi.zero(); u_phi.zero(); v_theta.zero(); u_theta.zero();

          for (int m = 0; m < NSEG; m++){

            matrix k_phi(3,1);
            k_phi(0) = filaments[n].vel_dir_phase[3*m];
            k_phi(1) = filaments[n].vel_dir_phase[3*m + 1];
            k_phi(2) = filaments[n].vel_dir_phase[3*m + 2];
            v_phi += k_phi;

            matrix k_theta(3,1);
            k_theta(0) = filaments[n].vel_dir_angle[3*m];
            k_theta(1) = filaments[n].vel_dir_angle[3*m + 1];
            k_theta(2) = filaments[n].vel_dir_angle[3*m + 2];
            v_theta += k_theta;

            matrix diff(3,1);
            diff(0) = filaments[n].segments[m].x[0] - x[0];
            diff(1) = filaments[n].segments[m].x[1] - x[1];
            diff(2) = filaments[n].segments[m].x[2] - x[2];

            u_phi += cross(diff, k_phi);
            u_theta += cross(diff, k_theta);

            c11 += dot(k_phi, k_phi);
            c22 += dot(k_theta, k_theta);
            c12 += dot(k_phi, k_theta);

          }          
          #if !DYNAMIC_SHAPE_ROTATION
            C_Ainv_B.add_to_block(0, 0, 3, 3, v_phi*transpose(v_phi)/c11);
            C_Ainv_B.add_to_block(3, 0, 3, 3, u_phi*transpose(v_phi)/c11);
            C_Ainv_B.add_to_block(3, 3, 3, 3, u_phi*transpose(u_phi)/c11);
          #else
            Real det = c11*c22 - c12*c12;
            cinv_11 = c22 / det;
            cinv_22 = c11 / det;
            cinv_12 = - c12 / det;

            matrix C_Ainv_B11 = v_phi*transpose(v_phi)*cinv_11 + v_phi*transpose(v_theta)*cinv_12
                            + v_theta*transpose(v_phi)*cinv_12 +  v_theta*transpose(v_theta)*cinv_22;
            matrix C_Ainv_B21 = u_phi*transpose(v_phi)*cinv_11 + u_phi*transpose(v_theta)*cinv_12
                            + u_theta*transpose(v_phi)*cinv_12 +  u_theta*transpose(v_theta)*cinv_22;
            matrix C_Ainv_B22 = u_phi*transpose(u_phi)*cinv_11 + u_phi*transpose(u_theta)*cinv_12
                            + u_theta*transpose(u_phi)*cinv_12 +  u_theta*transpose(u_theta)*cinv_22;
            C_Ainv_B.add_to_block(0, 0, 3, 3, C_Ainv_B11);
            C_Ainv_B.add_to_block(3, 0, 3, 3, C_Ainv_B21);
            C_Ainv_B.add_to_block(3, 3, 3, 3, C_Ainv_B22);

          #endif

        }

        // Use symmetry to finish off the new terms
        C_Ainv_B.set_block(0, 3, 3, 3, transpose(C_Ainv_B.get_block(3, 0, 3, 3)));

        // Finally, add on the new terms
        KTMinvK_inv -= seg_mob_fac*C_Ainv_B;

      #endif

      KTMinvK_inv = inverse(KTMinvK_inv);

    #endif

  }

#endif
