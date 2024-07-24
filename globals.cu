#include "config.hpp"
#include "globals.hpp"
#include <string>

// Global variables (CPU) first definition

// Values here are not meaningful.
int NSWIM = 1;
int NSEG = 1;
int NFIL = 16;
int NBLOB = 4000;
Real AR = 5;
Real TORSIONAL_SPRING_MAGNITUDE_FACTOR = 2.0;
Real GEN_FORCE_MAGNITUDE_FACTOR = 1.0;
int NTOTAL = (NSWIM*(NFIL_d*NSEG + NBLOB_d));
Real END_FORCE_MAGNITUDE = (DIMENSIONLESS_FORCE*KB/(DL*DL*NSEG*NSEG));
Real SEG_SEP = 2.6;
Real DL = SEG_SEP*RSEG;
Real SIM_LENGTH = 1.0;
Real PERIOD = 1.0;
Real DT = PERIOD/STEPS_PER_PERIOD;
int TOTAL_TIME_STEPS = 1;
Real TILT_ANGLE = 0.0;

std::string SIMULATION_DIR = "data/expr_sims/global/";
std::string SIMULATION_FILE = "cilia";


// Derived
Real AXIS_DIR_BODY_LENGTH = AR*44;
std::string SIMULATION_NAME = SIMULATION_DIR+SIMULATION_FILE;
std::string SIMULATION_CONFIG_NAME = SIMULATION_NAME + ".par";
std::string SIMULATION_BACKUP_NAME = SIMULATION_NAME + ".backup";
std::string SIMULATION_BODY_STATE_NAME = SIMULATION_NAME + "_body_states.dat"; // Blob states are recoverable from body states.
std::string SIMULATION_SEG_STATE_NAME = SIMULATION_NAME + "_seg_states.dat";
std::string SIMULATION_BODY_VEL_NAME = SIMULATION_NAME + "_body_vels.dat"; // Blob velocities are recoverable from body velocities.
std::string SIMULATION_SEG_VEL_NAME = SIMULATION_NAME + "_seg_vels.dat";
std::string SIMULATION_BLOB_FORCES_NAME = SIMULATION_NAME + "_blob_forces.dat"; // Body forces are recoverable from blob forces.
std::string SIMULATION_SEG_FORCES_NAME = SIMULATION_NAME + "_seg_forces.dat";
std::string SIMULATION_TIME_NAME = SIMULATION_NAME + "_time.dat";
std::string SIMULATION_TETHERLAM_NAME = SIMULATION_NAME + "_tether_force.dat";
std::string SIMULATION_TRUESTATE_NAME = SIMULATION_NAME + "_true_states.dat";

// Reading file names
std::string SIMULATION_FILPLACEMENT_NAME = "input/icosahedron/icosa_d3_N640.dat";
std::string SIMULATION_BLOBPLACEMENT_NAME = "input/icosahedron/icosa_d6_N40962.dat";
std::string CUFCM_CONFIG_FILE_NAME = "input/simulation_info_cilia";
std::string SIMULATION_ICSTATE_NAME = "input/psi.dat";
std::string SIMULATION_BODYSTATE_NAME = "input/bodystate.dat";