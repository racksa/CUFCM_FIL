#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <vector>
#include "swimmer.hpp"
#include <omp.h>
#include "cuda_functions.hpp"
#include "../general/util.hpp"
#include "../../config.hpp"



#if STOKES_DRAG_MOBILITY

  #include "stokes_drag_mobility_solver.hpp"
  typedef stokes_drag_mobility_solver chosen_mobility_solver;

#elif RPY_MOBILITY

  #include "rpy_mobility_solver.hpp"
  typedef rpy_mobility_solver chosen_mobility_solver;

#elif WEAKLY_COUPLED_FILAMENTS_RPY_MOBILITY

  #include "weakly_coupled_filaments_rpy_mobility_solver.hpp"
  typedef weakly_coupled_filaments_rpy_mobility_solver chosen_mobility_solver;

#elif UAMMD_FCM

  #include "uammd_fcm_mobility_solver.cuh"
  typedef uammd_fcm_mobility_solver chosen_mobility_solver;

#elif CUFCM

  #include "./mobility/fcm_mobility_solver.hpp"
  typedef fcm_mobility_solver chosen_mobility_solver;

#elif PAIRWISE_FCM

  #include "./mobility/pairwisefcm_mobility_solver.hpp"
  typedef pairwisefcm_mobility_solver chosen_mobility_solver;

#endif

#if !(PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

  #include "broyden_solver.hpp"

#endif

#include "../../config.hpp"
// #include "../../globals.hpp"


int main(int argc, char** argv){

  // Read global variables from .ini
  NSWIM = std::stoi(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "nswim"));
  NSEG = std::stoi(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "nseg"));
  NFIL = std::stoi(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "nfil"));
  NBLOB = std::stoi(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "nblob"));
  AR = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "ar"));
  SEG_SEP = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "seg_sep"));
  PERIOD = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "period"));
  SIM_LENGTH = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "sim_length"));
  TORSIONAL_SPRING_MAGNITUDE_FACTOR = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "spring_factor"));
  TILT_ANGLE = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "tilt_angle"));
  PAIR_DP = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "pair_dp"));
  WAVNUM = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "wavnum"));
  WAVNUM_DIA = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "wavnum_dia"));
  DIMENSIONLESS_FORCE = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "dimensionless_force"));
  FENE_MODEL = std::stoi(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "fene_model"));
  FORCE_NOISE_MAG = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "force_noise_mag"));
  OMEGA_SPREAD = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "omega_spread"));
  INDEX = std::stoi(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "index"));

  FIL_X_DIM = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "fil_x_dim"));
  FIL_SPACING = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "fil_spacing"));
  BLOB_X_DIM = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "blob_x_dim"));
  BLOB_SPACING = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "blob_spacing"));
  HEX_NUM = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "hex_num"));
  REV_RATIO = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "reverse_fil_direction_ratio"));
  TWOFIL_ANGLE = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "twofil_angle"));

  GEN_FORCE_MAGNITUDE_FACTOR = std::stof(data_from_ini(GLOBAL_FILE_NAME, "Parameters", "force_mag"));
  SIMULATION_DIR = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "simulation_dir");
  SIMULATION_FILE = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "simulation_file");

  SIMULATION_FILPLACEMENT_NAME = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "filplacement_file_name");
  SIMULATION_BLOBPLACEMENT_NAME = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "blobplacement_file_name");
  CUFCM_CONFIG_FILE_NAME = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "cufcm_config_file_name");
  SIMULATION_ICSTATE_NAME = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "simulation_icstate_name");
  SIMULATION_BODYSTATE_NAME = data_from_ini(GLOBAL_FILE_NAME, "Filenames", "simulation_bodystate_name");

  #if INFINITE_PLANE_WALL
    NSWIM = 1;
    NBLOB = 0;
  #endif

  #if WRITE_GENERALISED_FORCES
    NSWIM = 1;
    NFIL = 1;
    NBLOB = 0;
  #endif



  // Derive other global variables
  #ifdef BICILIA
    if (BICILIA || BICILIA_LONGT){
      NSEG_PER_CILIA = NSEG/2;
    }else{
      NSEG_PER_CILIA = NSEG;
    }
  #endif

  NPAIR = NFIL;
  if (PAIR==1){
    NPAIR = NFIL/2;
  }
  int NFIL_PER_PAIR = NFIL==0 ? 0 : NFIL/NPAIR;


  NTOTAL = (NSWIM*(NFIL*NSEG + NBLOB));
  AXIS_DIR_BODY_LENGTH = AR*FIL_LENGTH;
  END_FORCE_MAGNITUDE = (DIMENSIONLESS_FORCE*KB/(DL*DL*NSEG_PER_CILIA*NSEG_PER_CILIA));
  DL = SEG_SEP*RSEG;
  DT = PERIOD/STEPS_PER_PERIOD;
  TOTAL_TIME_STEPS = SIM_LENGTH*STEPS_PER_PERIOD+1;

  #if WRITE_GENERALISED_FORCES
    PERIOD = 1.0;
    DT = 1.0/STEPS_PER_PERIOD;
    TOTAL_TIME_STEPS = 300;
  #endif

  // Filenames 
  SIMULATION_NAME = SIMULATION_DIR + SIMULATION_FILE;
  SIMULATION_CONFIG_NAME = SIMULATION_NAME + ".par";
  SIMULATION_BACKUP_NAME = SIMULATION_NAME + ".backup";
  SIMULATION_BODY_STATE_NAME = SIMULATION_NAME + "_body_states.dat"; // Blob states are recoverable from body states.
  SIMULATION_SEG_STATE_NAME = SIMULATION_NAME + "_seg_states.dat";
  SIMULATION_BODY_VEL_NAME = SIMULATION_NAME + "_body_vels.dat"; // Blob velocities are recoverable from body velocities.
  SIMULATION_SEG_VEL_NAME = SIMULATION_NAME + "_seg_vels.dat";
  SIMULATION_BLOB_FORCES_NAME = SIMULATION_NAME + "_blob_forces.dat"; // Body forces are recoverable from blob forces.
  SIMULATION_SEG_FORCES_NAME = SIMULATION_NAME + "_seg_forces.dat";
  SIMULATION_TIME_NAME = SIMULATION_NAME + "_time.dat";
  SIMULATION_TETHERLAM_NAME = SIMULATION_NAME + "_tether_force.dat";
  SIMULATION_TRUESTATE_NAME = SIMULATION_NAME + "_true_states.dat";

  

  sync_var<<<1, 1>>>(NSWIM, NSEG, NFIL, NBLOB, END_FORCE_MAGNITUDE, SEG_SEP, DL);
  cudaDeviceSynchronize();

  float time_start;
  float solution_update_time;
  float jacobian_update_time;
  float hisolver_time;
  float eval_error_time;

  Real *data_from_file;

  #if READ_INITIAL_CONDITIONS_FROM_BACKUP

    std::cout << std::endl << std::endl << "Reading initial conditions from backup file..." << std::endl;

    std::ifstream input_file(INITIAL_CONDITIONS_FILE_NAME+std::string(".backup"));

    int nt_of_backup;

    #if PRESCRIBED_CILIA

      const int data_per_fil = 13;

    #else

      const int data_per_fil = 9 + 28*NSEG;

    #endif

    #if USE_BROYDEN_FOR_EVERYTHING

      const int data_per_swimmer = 19 + 9*NBLOB + NFIL*data_per_fil;

    #else

      const int data_per_swimmer = 19 + NFIL*data_per_fil;

    #endif

    const int backup_file_length = NSWIM*data_per_swimmer;

    data_from_file = new Real[backup_file_length];

    input_file >> nt_of_backup;

    for (int fpos = 0; fpos < backup_file_length; fpos++){

      input_file >> data_from_file[fpos];

    }

    std::cout << std::endl << "Finished reading from file!" << std::endl;

    input_file.close();

  #else

    const int data_per_swimmer = 0;

  #endif

  std::ofstream config_file(SIMULATION_CONFIG_NAME);

  config_file << NFIL << " " << "%% NFIL" << std::endl;
  config_file << NSEG << " " << "%% NSEG" << std::endl;
  config_file << NSWIM << " " << "%% NSWIM" << std::endl;
  config_file << NBLOB << " " << "%% NBLOB" << std::endl;
  config_file << MU << " " << "%% MU" << std::endl;
  config_file << RSEG << " " << "%% RSEG" << std::endl;
  config_file << DL << " " << "%% DL" << std::endl;
  config_file << FIL_LENGTH << " " << "%% FIL_LENGTH" << std::endl;
  config_file << RBLOB << " " << "%% RBLOB" << std::endl;
  config_file << KB << " " << "%% KB" << std::endl;
  config_file << KT << " " << "%% KT" << std::endl;
  config_file << LINEAR_SYSTEM_TOL << " " << "%% TOL" << std::endl;
  config_file << TOTAL_TIME_STEPS << " " << "%% TOTAL_TIME_STEPS" << std::endl;
  config_file << DT << " " << "%% DT" << std::endl;
  config_file << PLOT_FREQUENCY_IN_STEPS << " " << "%% PLOT_FREQUENCY_IN_STEPS" << std::endl;
  config_file << STEPS_PER_PERIOD << " " << "%% STEPS_PER_PERIOD" << std::endl;
  #if INSTABILITY_CILIA
    config_file << 0 << " " << "%% PRESCRIBED_CILIA" << std::endl;
    config_file << DIMENSIONLESS_FORCE << " " << "%% DIMENSIONLESS_FORCE" << std::endl;

  #elif PRESCRIBED_CILIA
    config_file << 1 << " " << "%% PRESCRIBED_CILIA" << std::endl;
    #if DYNAMIC_SHAPE_ROTATION
      config_file << TORSIONAL_SPRING_MAGNITUDE_FACTOR << " " << "%% TORSIONAL_SPRING_MAGNITUDE_FACTOR" << std::endl;

    #endif

  #elif GEOMETRIC_CILIA

    config_file << DRIVING_TORQUE_MAGNITUDE_RATIO << " " << "%% DRIVING_TORQUE_MAGNITUDE_RATIO" << std::endl;
    config_file << DEFAULT_BASE_TORQUE_MAGNITUDE << " " << "%% DEFAULT_BASE_TORQUE_MAGNITUDE" << std::endl;
    config_file << CRITICAL_ANGLE << " " << "%% CRITICAL_ANGLE" << std::endl;
    config_file << TRANSITION_STEPS << " " << "%% TRANSITION_STEPS" << std::endl;

  #elif CONSTANT_BASE_ROTATION

    config_file << BASE_ROTATION_RATE << " " << "%% BASE_ROTATION_RATE" << std::endl;

  #endif

  config_file.close();

  // Initialise the simulation
  chosen_mobility_solver mobility;
  mobility.initialise();
  mobility.write_generalised_forces();

  #if !(PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

    broyden_solver broyden;

  #endif

  std::vector<swimmer> swimmers(NSWIM);

  for (int n = 0; n < NSWIM; n++){
    swimmers[n].initial_setup(n, &data_from_file[n*data_per_swimmer],
                                        &mobility.x_segs_host[3*n*NFIL*NSEG],
                                        &mobility.f_segs_host[6*n*NFIL*NSEG],
                                        &mobility.f_blobs_host[3*n*NBLOB]);
  }

  #if READ_INITIAL_CONDITIONS_FROM_BACKUP

    delete[] data_from_file;

  #endif

  // Swimmers are all identical
  swimmers[0].write_reference_positions();
  

  #if !(PRESCRIBED_BODY_VELOCITIES || PRESCRIBED_CILIA || USE_BROYDEN_FOR_EVERYTHING)

    mobility.make_body_reference_matrices(swimmers);

    for (int n = 0; n < NSWIM; n++){

      swimmers[n].body_mobility_reference = mobility.body_mobility_reference;

    }

  #endif

  #if INFINITE_PLANE_WALL

    std::cout << std::endl;
    std::cout << "Simulating " << NFIL << " filaments on an infinite no-slip wall, with each filament comprised of " << NSEG << " segments." << std::endl;
    std::cout << std::endl;

  #else

    std::cout << std::endl;
    std::cout << "Simulating " << NSWIM << " swimmers, each having a rigid body resolved using " << NBLOB << " 'blobs'." << std::endl;
    std::cout << "Attached to each rigid body are " << NPAIR << " pairs of filaments, each pair comprised of " << NFIL_PER_PAIR << " filaments each with " << NSEG << " segments and length " << FIL_LENGTH << std::endl;
    std::cout << std::endl;

  #endif

  #if PRESCRIBED_CILIA

    #if WRITE_GENERALISED_FORCES

      std::ofstream phase_generalised_force_file(reference_phase_generalised_force_file_name());

      phase_generalised_force_file << TOTAL_TIME_STEPS << " ";
      phase_generalised_force_file << std::scientific << std::setprecision(15);

      std::ofstream angle_generalised_force_file(reference_angle_generalised_force_file_name());

      angle_generalised_force_file << TOTAL_TIME_STEPS << " ";
      angle_generalised_force_file << std::scientific << std::setprecision(15);

      std::ofstream s_values_file(reference_s_values_file_name());

      s_values_file << TOTAL_TIME_STEPS << " " << NSEG_PER_CILIA << " ";
      s_values_file << std::scientific << std::setprecision(15);

      s_values_file.close(); // Unlike the force file, we close this one. We'll open it again with append permissions inside the filament class.

    #elif FIT_TO_DATA_BEAT // i.e. if (FIT_TO_DATA_BEAT && !WRITE_GENERALISED_FORCES)

      std::ifstream s_values_file(reference_s_values_file_name());

      if (!s_values_file.good()){

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Required reference file '" << reference_s_values_file_name() << "' for s-values was not found." << std::endl;
        std::cout << "Run an appropriate simulation with the WRITE_GENERALISED_FORCES option enabled." << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;

        std::exit(-1);

      }

      int ref_num_steps, ref_num_segs;
      s_values_file >> ref_num_steps;
      s_values_file >> ref_num_segs;

      matrix s_values_ref(ref_num_steps, ref_num_segs); // I want this to be stored at a 'global' level and just referenced by each filament, but there's no obvious choice of class it should belong to.

      for (int ii = 0; ii < ref_num_steps; ii++){
        for (int jj = 0; jj < ref_num_segs; jj++){

          s_values_file >> s_values_ref(ii,jj);

        }
      }

      s_values_file.close();

      // The swimmers (and hence filaments) have already been set up, so we can just give them the reference now.
      for (int ii = 0; ii < NSWIM; ii++){
        for (int jj = 0; jj < NFIL; jj++){

          swimmers[ii].filaments[jj].s_to_use_ref_ptr = &s_values_ref;

        }
      }

    #endif

  #endif

  // If continuing from backup file...
  #if INITIAL_CONDITIONS_TYPE==1

    #if (PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

      const int nt_start = nt_of_backup + 1; // Explicit description

    #else

      const int nt_start = nt_of_backup; // Implicit description

    #endif

  #else

    const int nt_start = 0;

  #endif

  // Begin time stepping
  for (int nt = nt_start; nt < TOTAL_TIME_STEPS; nt++) {

    for (int i = 0; i < NSWIM; i++) {

      swimmers[i].initial_guess(nt);

      swimmers[i].forces_and_torques(nt, i);

    }

    int num_gmres_iterations;
    mobility.compute_velocities(swimmers, num_gmres_iterations, nt);

    #if (PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

      #if WRITE_GENERALISED_FORCES

        Real phase_generalised_force = 0.0;
        Real angle_generalised_force = 0.0;

        for (int n = 0; n < NSEG; n++){

          phase_generalised_force += swimmers[0].filaments[0].vel_dir_phase[3*n]*mobility.f_segs_host[6*n];
          phase_generalised_force += swimmers[0].filaments[0].vel_dir_phase[3*n + 1]*mobility.f_segs_host[6*n + 1];
          phase_generalised_force += swimmers[0].filaments[0].vel_dir_phase[3*n + 2]*mobility.f_segs_host[6*n + 2];

          angle_generalised_force += swimmers[0].filaments[0].vel_dir_angle[3*n]*mobility.f_segs_host[6*n];
          angle_generalised_force += swimmers[0].filaments[0].vel_dir_angle[3*n + 1]*mobility.f_segs_host[6*n + 1];
          angle_generalised_force += swimmers[0].filaments[0].vel_dir_angle[3*n + 2]*mobility.f_segs_host[6*n + 2];

        }

        phase_generalised_force_file << phase_generalised_force << " ";
        angle_generalised_force_file << angle_generalised_force << " ";

      #endif

    #else

      bool error_is_too_large = mobility.compute_errors(broyden.error, swimmers, nt);

      if (error_is_too_large){

        for (int i = 0; i < NSWIM; i++) {

          swimmers[i].prepare_jacobian_inv(nt);

        }

      }

      broyden.iter = 0;
      
      solution_update_time = 0.0;
      hisolver_time = 0.0;
      eval_error_time = 0.0;
      jacobian_update_time = 0.0;

      while (error_is_too_large && (broyden.iter < MAX_BROYDEN_ITER)){

        #if DISPLAYTIME && CUFCM
          cudaDeviceSynchronize(); time_start = get_time();
        #endif

        broyden.find_update(swimmers, nt);

        
        for (int i = 0; i < NSWIM; i++) {

          #if INFINITE_PLANE_WALL

            const int per_body = 6*NFIL*NSEG;

          #elif USE_BROYDEN_FOR_EVERYTHING

            const int per_body = 6*NFIL*NSEG + 3*NBLOB + 6;

          #else

            const int per_body = 6*NFIL*NSEG + 6;

          #endif

          // update lie algebra element
          // robot_arm to update positions
          swimmers[i].update(&broyden.update.data[i*per_body]);

          swimmers[i].forces_and_torques(nt, i);

        }

        #if DISPLAYTIME && CUFCM
          if(nt%10==0){ cudaDeviceSynchronize(); solution_update_time += (get_time() - time_start); time_start = get_time();}
        #endif

        mobility.compute_velocities(swimmers, num_gmres_iterations, nt);

        #if DISPLAYTIME && CUFCM
          if(nt%10==0){ cudaDeviceSynchronize(); hisolver_time += (get_time() - time_start); time_start = get_time();}
        #endif

        #if CUFCM
          if(nt%10==0){
            printf("Checking overlap\n");
            mobility.cufcm_solver->check_overlap();
          }
        #endif

        error_is_too_large = mobility.compute_errors(broyden.new_error, swimmers, nt);

        #if DISPLAYTIME && CUFCM
          if(nt%10==0){ cudaDeviceSynchronize(); eval_error_time += (get_time() - time_start); time_start = get_time();}
        #endif

        if (!broyden.new_error.is_finite()){

          std::cout << DELETE_CURRENT_LINE << std::flush;
          std::cout << "\t\tBroyden's method diverged after " << broyden.iter+1 << " iterations during step " << nt+1 << "." << std::endl;
          #if CUFCM
            
            std::ofstream seg_state_file(SIMULATION_SEG_STATE_NAME, std::ios::app);
            std::ofstream body_state_file(SIMULATION_BODY_STATE_NAME, std::ios::app);
            // swimmers[0].write_data(seg_state_file, body_state_file);

            mobility.copy_to_fcm();
            printf("Checking overlap\n");
            mobility.cufcm_solver->check_overlap();
            mobility.cufcm_solver->write_data_call();
            mobility.write_repulsion();
          #endif
          return 0;

        }

        broyden.end_of_iter(swimmers, nt, nt_start, error_is_too_large);

        #if DISPLAYTIME && CUFCM
          if(nt%10==0){ cudaDeviceSynchronize(); jacobian_update_time += (get_time() - time_start); time_start = get_time();}
        #endif

        std::cout << DELETE_CURRENT_LINE << std::flush;
        std::cout << "Step " << nt+1 << ": Completed Broyden iteration " << broyden.iter;
        #if !(INFINITE_PLANE_WALL || USE_BROYDEN_FOR_EVERYTHING)
          std::cout << " in " << num_gmres_iterations << " iterations of the linear system solver." << "\r";
        #endif        
        std::cout << std::flush;

      }

      #if DISPLAYTIME && CUFCM
        if(nt%10==0){
            float total_time = solution_update_time+hisolver_time+eval_error_time+jacobian_update_time;
            std::ofstream time_file(SIMULATION_TIME_NAME, std::ios::app);
            time_file << nt << " " << broyden.iter << " ";
            time_file << std::scientific << std::setprecision(6);
            
            time_file<<(solution_update_time/(broyden.iter+1))<<" ";
            time_file<<(hisolver_time/(broyden.iter+1))<<" ";
            time_file<<(eval_error_time/(broyden.iter+1))<<" "; 
            time_file<<(jacobian_update_time/(broyden.iter+1))<<" ";
            time_file<<(total_time/(broyden.iter+1))<<" ";

            time_file << std::endl;
            time_file.close();
          }
        #endif
    #endif

    for (int i = 0; i < NSWIM; i++) {

      swimmers[i].end_of_step(nt);

    }

    #if CUFCM
      if(nt%10==0){
        std::cout << "Checking overlap";
        std::cout << DELETE_CURRENT_LINE << std::flush;
        
        mobility.cufcm_solver->check_overlap();

        mobility.cufcm_solver->prompt_time();
      }
    #endif

    #if (PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

      std::cout << "Completed step " << nt+1 << "/" << TOTAL_TIME_STEPS << " in " << num_gmres_iterations << " iterations of the linear system solver." << std::endl;

    #else

      if (error_is_too_large){

        std::cout << DELETE_CURRENT_LINE << std::flush;
        std::cout << "Broyden's method failed to converge within " << MAX_BROYDEN_ITER << " iterations during step " << nt+1 << "." << std::endl;
        #if CUFCM
          

          std::ofstream seg_state_file(SIMULATION_SEG_STATE_NAME, std::ios::app);
          std::ofstream body_state_file(SIMULATION_BODY_STATE_NAME, std::ios::app);
          // swimmers[0].write_data(seg_state_file, body_state_file);

          mobility.copy_to_fcm();
          printf("Checking overlap\n");
          mobility.cufcm_solver->check_overlap();
          mobility.cufcm_solver->write_data_call();
          mobility.write_repulsion();
        #endif
        return 0;

      } else {

        std::cout << DELETE_CURRENT_LINE << std::flush;
        std::cout << "Completed step " << nt+1 << "/" << TOTAL_TIME_STEPS << " in " << broyden.iter << " Broyden iterations (max = " << broyden.max_iter << ", avg. = " << broyden.avg_iter << ")." << std::endl;

      }

    #endif

    if (nt%PLOT_FREQUENCY_IN_STEPS == 0){

      #if (PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

        const int save_step = nt; // Explicit description

      #else

        const int save_step = nt+1; // Implicit description

      #endif

      std::ofstream seg_state_file(SIMULATION_SEG_STATE_NAME, std::ios::app);
      seg_state_file << save_step << " ";
      seg_state_file << std::scientific << std::setprecision(OUTPUT_DIGIT);

      std::ofstream body_state_file(SIMULATION_BODY_STATE_NAME, std::ios::app);
      body_state_file << save_step << " ";
      body_state_file << std::scientific << std::setprecision(OUTPUT_DIGIT);

      for (int n = 0; n < NSWIM; n++){

        swimmers[n].write_data(seg_state_file, body_state_file);

      }

      seg_state_file << std::endl;
      seg_state_file.close();

      body_state_file << std::endl;
      body_state_file.close();

      mobility.write_data(nt, swimmers); // Writes all velocity and force data.

      // std::ofstream backup_file(SIMULATION_BACKUP_NAME);
      // backup_file << save_step << " ";
      // backup_file << std::scientific << std::setprecision(OUTPUT_DIGIT);

      // for (int n = 0; n < NSWIM; n++){

      //   swimmers[n].write_backup(backup_file);

      // }

      // backup_file << std::endl;
      // backup_file.close();

      #if PRESCRIBED_CILIA

        // std::ofstream fil_phase_file(SIMULATION_NAME+std::string("_filament_phases.dat"), std::ios::app);
        // fil_phase_file << save_step << " ";
        // fil_phase_file << std::scientific << std::setprecision(OUTPUT_DIGIT);

        std::ofstream true_states_file(SIMULATION_TRUESTATE_NAME, std::ios::app);
        true_states_file << std::scientific << std::setprecision(OUTPUT_DIGIT);
        true_states_file << save_step << " " << PERIOD << " ";

        for (int n = 0; n < NSWIM; n++){

          for (int m = 0; m < NFIL; m++){

            // fil_phase_file << swimmers[n].filaments[m].phase << " ";
            true_states_file << swimmers[n].filaments[m].phase << " ";

          }

        }

        // fil_phase_file << std::endl;
        // fil_phase_file.close();

        #if DYNAMIC_SHAPE_ROTATION

          // std::ofstream fil_angle_file(SIMULATION_NAME+std::string("_filament_shape_rotation_angles.dat"), std::ios::app);
          // fil_angle_file << save_step << " ";
          // fil_angle_file << std::scientific << std::setprecision(OUTPUT_DIGIT);

          for (int n = 0; n < NSWIM; n++){

            for (int m = 0; m < NFIL; m++){

              // fil_angle_file << swimmers[n].filaments[m].shape_rotation_angle << " ";
              true_states_file << swimmers[n].filaments[m].shape_rotation_angle << " ";

            }

          }

          // fil_angle_file << std::endl;
          // fil_angle_file.close();

        #endif

        true_states_file << std::endl;
        true_states_file.close();

      #endif

    }

    #if (PRESCRIBED_CILIA || NO_CILIA_SQUIRMER)

      // Explicit time integration for these simulation types.
      // This update occurs AFTER we write to file to preserve the explicit description in the data files.
      for (int n = 0; n < NSWIM; n++){

        swimmers[n].body.x[0] += DT*mobility.v_bodies(6*n);
        swimmers[n].body.x[1] += DT*mobility.v_bodies(6*n + 1);
        swimmers[n].body.x[2] += DT*mobility.v_bodies(6*n + 2);

        swimmers[n].body.u[0] = DT*mobility.v_bodies(6*n + 3);
        swimmers[n].body.u[1] = DT*mobility.v_bodies(6*n + 4);
        swimmers[n].body.u[2] = DT*mobility.v_bodies(6*n + 5);
        swimmers[n].body.q = lie_exp(swimmers[n].body.u)*swimmers[n].body.qm1;

      }

    #endif

  }

  mobility.finalise();

  #if WRITE_GENERALISED_FORCES

    phase_generalised_force_file << std::endl;
    phase_generalised_force_file.close();

    angle_generalised_force_file << std::endl;
    angle_generalised_force_file.close();

  #endif

  return 0;

};
