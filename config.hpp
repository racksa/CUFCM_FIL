// config.hpp
#include <string>

// ============================================================================
// TABLE OF CONTENTS
//   [SEC: FILE NAMES]       - output file path extern declarations
//   [SEC: PRECISION]        - float vs double toggle, Real/Integer typedefs
//   [SEC: SIMULATION TYPE]  - cilia, body/surface, mobility, motion, IC types
//   [SEC: PHYSICAL PARAMS]  - KB, KT, MU, RSEG, RBLOB, geometry externs
//   [SEC: COMPUTATIONAL]    - solver tolerances, iterations, time-stepping
//   [SEC: TYPE ALIASES]     - boolean shorthand macros (do not edit)
//   [SEC: COMPUTED PARAMS]  - derived values and validation (do not edit)
// ============================================================================

// ============================================================================
// Include guard
// ============================================================================
#ifndef MY_CONFIG_HEADER_INCLUDED
#define MY_CONFIG_HEADER_INCLUDED

// ============================================================================
// [SEC: FILE NAMES]
// ============================================================================

extern std::string SIMULATION_DIR;
extern std::string SIMULATION_FILE;
extern std::string SIMULATION_NAME;

extern std::string SIMULATION_CONFIG_NAME;
extern std::string SIMULATION_BACKUP_NAME;
extern std::string SIMULATION_BODY_STATE_NAME; // Blob states are recoverable from body states.
extern std::string SIMULATION_SEG_STATE_NAME;
extern std::string SIMULATION_BODY_VEL_NAME;   // Blob velocities are recoverable from body velocities.
extern std::string SIMULATION_SEG_VEL_NAME;
extern std::string SIMULATION_BLOB_FORCES_NAME; // Body forces are recoverable from blob forces.
extern std::string SIMULATION_SEG_FORCES_NAME;
extern std::string SIMULATION_TIME_NAME;
extern std::string SIMULATION_TETHERLAM_NAME;
extern std::string SIMULATION_TRUESTATE_NAME;

extern std::string SIMULATION_FILPLACEMENT_NAME;
extern std::string SIMULATION_BLOBPLACEMENT_NAME;
extern std::string SIMULATION_ICSTATE_NAME;
extern std::string SIMULATION_BODYSTATE_NAME;
extern std::string CUFCM_CONFIG_FILE_NAME;

#define GLOBAL_FILE_NAME "input/globals.ini"

// ============================================================================
// [SEC: PRECISION]
// ============================================================================

#define FIL_USE_DOUBLE_PRECISION false

#if FIL_USE_DOUBLE_PRECISION
    typedef double Real;
    typedef long Integer;
    #define myfil_sqrt   sqrt
    #define myfil_rint   rint
    #define myfil_exp    exp
    #define myfil_floor  floor
    #define myfil_ceil   ceil
    #define myfil_fmod   fmod
    #define myfil_getrf_ dgetrf_
    #define myfil_getri_ dgetri_
    #define myfil_gemm_  dgemm_
    #define myfil_cos    cos
    #define myfil_sin    sin
    #define OUTPUT_DIGIT 15
    #define my_sqrt      sqrt
#else
    typedef float Real;
    typedef int Integer;
    #define myfil_sqrt   sqrtf
    #define myfil_rint   rintf
    #define myfil_exp    expf
    #define myfil_floor  floorf
    #define myfil_ceil   ceilf
    #define myfil_fmod   fmodf
    #define myfil_getrf_ sgetrf_
    #define myfil_getri_ sgetri_
    #define myfil_gemm_  sgemm_
    #define myfil_cos    cosf
    #define myfil_sin    sinf
    #define OUTPUT_DIGIT 7
    #define my_sqrt      sqrtf
#endif

// ============================================================================
// [SEC: SIMULATION TYPE]
// ============================================================================

// --- Cilia type ----------------------------------------------------------
#define CILIA_TYPE 3
// Valid options:
// 0 = Instability-driven cilia.
// 1 = Geometrically-switching cilia (partially implemented).
// 2 = Constant base rotation (partially implemented).
// 3 = Cilia follow a prescribed sequence of shapes.
// 4 = Squirmer-type simulation (no filaments; slip velocity set in mobility solver).

#define PAIR 0
// Sub-type for prescribed cilia motion.
// Enables filaments seeded as pairs with different frequencies per filament.
// Use 0 for bicilia.

#if CILIA_TYPE==0

  #define CILIA_IC_TYPE 2
  // Valid options:
  // 0 = All cilia have identical planar perturbations.
  // 1 = All cilia have identical out-of-plane perturbations.
  // 2 = Cilia have random planar perturbations.
  // 3 = Cilia have random out-of-plane perturbations.

#elif CILIA_TYPE==3

  #define SHAPE_SEQUENCE 1
  // Valid options:
  // 0 = 'Build-a-beat'.
  // 1 = Fulford and Blake beat (mammalian airway cilia).
  // 2 = Coral larvae beat pattern.
  // 3 = Volvox beat.
  // 4 = Original Fulford and Blake beat, <L>=0.975.
  // 5 = Bi-cilia, fixed phase difference.
  // 6 = Bi-cilia long T, variable phase difference.
  // 7 = Fulford and Blake beat with no-wall generalised force.

  #define DYNAMIC_PHASE_EVOLUTION false
  // If true, cilia phase speeds are solved for as part of the dynamics.
  // Requires a prior reference simulation with WRITE_GENERALISED_FORCES=true.

  #define DYNAMIC_SHAPE_ROTATION false
  // If true, cilia can tip backwards or forwards in their beat planes.

  #define WRITE_GENERALISED_FORCES true
  // If true, saves generalised forces for use as reference values.
  // NOTE: Overwrites any existing reference files.

  #define CILIA_IC_TYPE 5
  // Valid options:
  // 0 = All cilia start in-phase with phase 0.
  // 1 = Random initial phase (deprecated).
  // 2 = Metachronal wave (deprecated).
  // 3 = Ishikawa MCW (deprecated).
  // 5 = Read from file (default).
  // 6 = Prescribed MCW from globals.ini.

#endif

// --- Body / surface type -------------------------------------------------
#define BODY_OR_SURFACE_TYPE 0
// Valid options:
// 0 = Infinite plane wall at z=0 (only compatible with RPY).
// 1 = Deformed planes with 2 principal curvatures (partially implemented).
// 2 = Surface-of-revolution bodies.
// 3 = Toroidal bodies (partially implemented).
// 4 = Rigid rod.
// 5 = Filament on rigid wall.

#if BODY_OR_SURFACE_TYPE==0

  #define INFINITE_PLANE_WALL_SEEDING_TYPE 1
  // Valid options:
  // 0 = Rectangular grid.
  // 1 = Hexagonal grid.
  // 2 = Rectangular grid, cuFCM only (deprecated).
  // 3 = Lattice, cuFCM only (deprecated).
  // 4 = Two-filament pair configuration.

  // Force rectangular seeding when writing generalised forces.
  #if WRITE_GENERALISED_FORCES
    #undef  INFINITE_PLANE_WALL_SEEDING_TYPE
    #define INFINITE_PLANE_WALL_SEEDING_TYPE 0
  #endif

  // Define one lattice size; leave the other blank for automatic calculation.
  // Leave both blank for a regular lattice.
  #define FIL_LATTICE_X_NUM     FIL_X_DIM
  #define FIL_LATTICE_Y_NUM
  #define FIL_LATTICE_Y_SPACING FIL_SPACING // For prescribed-shape cilia: beat-wise separation.
  #define FIL_LATTICE_X_SPACING FIL_X_SPACING // Zero → automatic (regular or hexagonal default).

#elif BODY_OR_SURFACE_TYPE==2 or BODY_OR_SURFACE_TYPE==4 or BODY_OR_SURFACE_TYPE==5

  #define SEEDING_TYPE 7
  // Valid options:
  // 0 = Evenly distributed over surface.
  // 1 = Equatorial band.
  // 2 = Platynereis-style (equatorial band + small rear ring).
  // 3 = Hexagonal grid (rigid-body plane).
  // 4 = Meridian seeding.
  // 5 = Icosa seeding.
  // 6 = Mismatched seeding.
  // 7 = Evenly distributed with potential at poles.
  // 8 = Equal-area centric seeding (pizza-like rigid-body plane).
  // 9 = Read from files.

  #if BODY_OR_SURFACE_TYPE==5
    #define FOURIER_DIR          "input/rigidwall_seeding/"
    #define GENERATRIX_FILE_NAME FOURIER_DIR "rigidwall"
  #else
    #if CILIA_TYPE==4
      #define FOURIER_DIR "input/fourier_modes_resolution_study/"
    #elif PAIR==1
      #define FOURIER_DIR "input/fourier_modes_pair/"
    #else
      #define FOURIER_DIR "input/fourier_modes/"
    #endif
    #define GENERATRIX_FILE_NAME FOURIER_DIR "sphere"
  #endif

#endif

// --- Mobility type -------------------------------------------------------
#define MOBILITY_TYPE 1
// Valid options:
// 0 = Basic Stokes drag (no hydrodynamic interactions).
// 1 = Rotne-Prager-Yamakawa (RPY), with Swan-Brady wall corrections if applicable.
// 2 = Weakly-coupled-filaments RPY (inter-filament interactions approximated).
// 3 = Force Coupling Method (FCM) via UAMMD.
// 4 = cuFCM.
// 5 = Pairwise FCM.

// --- Body motion ---------------------------------------------------------
#define BODY_VELOCITY_TYPE 1
// 0 = Free to swim.
// 1 = Prescribed velocities.
// 2 = Prescribed rotation only.

#define PRESCRIBED_BODY_VELOCITIES (BODY_VELOCITY_TYPE == 1)
#define PRESCRIBED_BODY_ROTATION   (BODY_VELOCITY_TYPE == 2)

// --- Initial conditions --------------------------------------------------
#define INITIAL_CONDITIONS_TYPE 0
// Valid options:
// 0 = Default.
// 1 = Resume from backup file (DT must match; not checked automatically).
// 2 = Fresh start from backup file (uses saved state but resets t=0).
// #define INITIAL_CONDITIONS_FILE_NAME SIMULATION_NAME

// --- Output --------------------------------------------------------------
#define OUTPUT_FORCES true

// ============================================================================
// [SEC: PHYSICAL PARAMS]
// ============================================================================

extern int  NSWIM;
extern int  NSEG;
extern int  NSEG_PER_CILIA;
extern int  NFIL;
extern int  NPAIR;
extern int  NBLOB;
extern Real AR;
extern Real AXIS_DIR_BODY_LENGTH;
extern Real TORSIONAL_SPRING_MAGNITUDE_FACTOR; // Pre-multiplies mean generalised driving force to give rotational spring constant.
extern Real GEN_FORCE_MAGNITUDE_FACTOR;
extern int  NTOTAL;
extern Real END_FORCE_MAGNITUDE;
extern Real SEG_SEP;
extern Real DL;
extern Real SIM_LENGTH;
extern Real PERIOD;
extern Real DT;
extern int  TOTAL_TIME_STEPS;
extern Real TILT_ANGLE;
extern Real PAIR_DP;
extern Real WAVNUM;
extern Real WAVNUM_DIA;
extern Real DIMENSIONLESS_FORCE;
extern int  FENE_MODEL;
extern Real FORCE_NOISE_MAG;
extern Real OMEGA_SPREAD;
extern int  INDEX;

extern Real FIL_X_DIM;
extern Real FIL_Y_DIM;
extern Real FIL_SPACING;
extern Real FIL_X_SPACING;
extern Real BLOB_X_DIM;
extern Real BLOB_Y_DIM;
extern Real BLOB_SPACING;
extern Real HEX_NUM;
extern Real TWOFIL_ANGLE;
extern Real REV_RATIO;

#define MU    1.0 // Fluid viscosity.
#define RSEG  1.0 // Segment radius.
#define RBLOB 1.0 // Surface blob radius.

#define KB 1800.0 // Bending modulus.
#define KT 1800.0 // Twist modulus.

#if CILIA_TYPE==1
  #define DRIVING_TORQUE_MAGNITUDE_RATIO 3.0      // Ratio of fast-stroke to slow-stroke driving torque.
  #define DEFAULT_BASE_TORQUE_MAGNITUDE  (0.1*KB) // Driving torque magnitude in the fast stroke.
  #define CRITICAL_ANGLE                 (0.4*PI) // Angle at which the cilia changes stroke.
#elif CILIA_TYPE==2
  #define BASE_ROTATION_RATE 0.1
#endif

// ============================================================================
// [SEC: COMPUTATIONAL]
// ============================================================================

// Threads per block for kernel execution. Should be a multiple of 32 (warp size).
#define THREADS_PER_BLOCK 64

// Dampens Broyden updates: 1.0 = standard, <1.0 reduces over-adjustment.
// 0.4 works well with GMRES; 0.1 for Broyden-only simulations.
#define JACOBIAN_CONFIDENCE_FACTOR 0.1

#define MAX_BROYDEN_ITER 400
#define TOL              1e-4

#define SOLVER_TYPE 1
// Valid options:
// 0 = Broyden's method for everything (linear system embedded in Broyden solve).
// 1 = GMRES to solve the linear system at each Broyden iteration.

#if SOLVER_TYPE==1
  // Right preconditioning (default): GMRES error equals original system error.
  // Left preconditioning: set to false.
  #define USE_RIGHT_PRECON true
#endif

#define MAX_LINEAR_SYSTEM_ITER 350
#define LINEAR_SYSTEM_TOL      1e-4

#define NUM_EULER_STEPS 1 // Number of backwards-Euler steps before switching to BDF2.

#if CILIA_TYPE==1
  #define DT                      2.0
  #define PLOT_FREQUENCY_IN_STEPS 10
  #define TRANSITION_STEPS        21  // Time-steps to ramp driving torque between strokes (>=1).
#else
  #define STEPS_PER_PERIOD 300
  #define SAVES_PER_PERIOD 30
#endif

#define BASE_HEIGHT_ABOVE_SURFACE   (0.5*DL)
#define BASE_HEIGHT_ABOVE_SURFACE_d (0.5*DL_d)

// ============================================================================
// [SEC: TYPE ALIASES]
// Boolean shorthands derived from the type flags above. Do not edit.
// ============================================================================

#define GEOMETRIC_CILIA        (CILIA_TYPE==1)
#define INSTABILITY_CILIA      (CILIA_TYPE==0)
#define CONSTANT_BASE_ROTATION (CILIA_TYPE==2)
#define PRESCRIBED_CILIA       (CILIA_TYPE==3)
#define NO_CILIA_SQUIRMER      (CILIA_TYPE==4)

#define INFINITE_PLANE_WALL          (BODY_OR_SURFACE_TYPE==0)
#define SADDLE_BODIES                (BODY_OR_SURFACE_TYPE==1)
#define SURFACE_OF_REVOLUTION_BODIES (BODY_OR_SURFACE_TYPE==2)
#define TORUS_BODIES                 (BODY_OR_SURFACE_TYPE==3)
#define ROD                          (BODY_OR_SURFACE_TYPE==4)
#define RIGIDWALL                    (BODY_OR_SURFACE_TYPE==5)

#define STOKES_DRAG_MOBILITY                  (MOBILITY_TYPE==0)
#define RPY_MOBILITY                          (MOBILITY_TYPE==1)
#define WEAKLY_COUPLED_FILAMENTS_RPY_MOBILITY (MOBILITY_TYPE==2)
#define UAMMD_FCM                             (MOBILITY_TYPE==3)
#define CUFCM                                 (MOBILITY_TYPE==4)
#define PAIRWISE_FCM                          (MOBILITY_TYPE==5)

#define USE_BROYDEN_FOR_EVERYTHING  (SOLVER_TYPE==0)
#define USE_GMRES_FOR_LINEAR_SYSTEM (SOLVER_TYPE==1)

#define READ_INITIAL_CONDITIONS_FROM_BACKUP (INITIAL_CONDITIONS_TYPE != 0)

#if INFINITE_PLANE_WALL
  #define RECTANGULAR_SEEDING     (INFINITE_PLANE_WALL_SEEDING_TYPE==0)
  #define HEXAGONAL_SEEDING       (INFINITE_PLANE_WALL_SEEDING_TYPE==1)
  #define FCM_RECTANGULAR_SEEDING (INFINITE_PLANE_WALL_SEEDING_TYPE==2)
  #define FCM_LATTICE_SEEDING     (INFINITE_PLANE_WALL_SEEDING_TYPE==3)
  #define TWOFIL_SEEDING          (INFINITE_PLANE_WALL_SEEDING_TYPE==4)
#endif

#if SURFACE_OF_REVOLUTION_BODIES or RIGIDWALL
  #define UNIFORM_SEEDING        (SEEDING_TYPE==0)
  #define EQUATORIAL_SEEDING     (SEEDING_TYPE==1)
  #define PLATY_SEEDING          (SEEDING_TYPE==2)
  #define HEXAGONAL_WALL_SEEDING (SEEDING_TYPE==3)
  #define MERIDIAN_SEEDING       (SEEDING_TYPE==4)
  #define ICOSA_SEEDING          (SEEDING_TYPE==5)
  #define MISMATCH_SEEDING       (SEEDING_TYPE==6)
  #define UNIFORM_SEEDING_POLE   (SEEDING_TYPE==7)
  #define CENTRIC_WALL_SEEDING   (SEEDING_TYPE==8)
#endif

#if PRESCRIBED_CILIA
  #define BUILD_A_BEAT                    (SHAPE_SEQUENCE==0)
  #define FIT_TO_DATA_BEAT                (SHAPE_SEQUENCE != 0)
  #define FULFORD_AND_BLAKE_BEAT          (SHAPE_SEQUENCE==1)
  #define CORAL_LARVAE_BEAT               (SHAPE_SEQUENCE==2)
  #define VOLVOX_BEAT                     (SHAPE_SEQUENCE==3)
  #define FULFORD_AND_BLAKE_BEAT_ORIGINAL (SHAPE_SEQUENCE==4)
  #define BICILIA                         (SHAPE_SEQUENCE==5) // deprecated - use PAIR instead
  #define BICILIA_LONGT                   (SHAPE_SEQUENCE==6) // deprecated - use PAIR instead
  #define FULFORD_AND_BLAKE_BEAT_NO_WALL  (SHAPE_SEQUENCE==7)
#endif

#define PI 3.14159265358979323846264338327950288

#define DELETE_CURRENT_LINE "                                                                                                               " << "\r"

#define DEFINED_BUT_EMPTY(VAR) ((~(~VAR + 0) == 0) && (~(~VAR + 1) == 1)) // Macro magic...

// ============================================================================
// [SEC: COMPUTED PARAMS]
// Derived values and validation. Do not edit.
// ============================================================================

#if USE_BROYDEN_FOR_EVERYTHING
  #define NBROY (3*NSWIM*(NBLOB + 2*(NFIL*NSEG + 1)))
#else
  #define NBROY (6*(NSWIM*NFIL*NSEG + NSWIM))
#endif

#if !GEOMETRIC_CILIA
  #define PLOT_FREQUENCY_IN_STEPS (STEPS_PER_PERIOD/SAVES_PER_PERIOD)
#endif

#if INSTABILITY_CILIA

  // #define END_FORCE_MAGNITUDE (DIMENSIONLESS_FORCE*KB/(DL*DL*NSEG*NSEG))
  #define REPULSIVE_FORCE_FACTOR 2.0 // How much stronger is the barrier force than the driving force.
  // #define DT (36.3833/STEPS_PER_PERIOD) // Based on the period of a DIMENSIONLESS_FORCE=220 filament above a no-slip wall.

#elif CONSTANT_BASE_ROTATION

  #define DT (2.0*PI/(BASE_ROTATION_RATE*STEPS_PER_PERIOD))
  #define REPULSIVE_FORCE_FACTOR 1.0
  #define END_FORCE_MAGNITUDE (0.1*KB) // Not a real end force; used only to define the repulsion scale.

#elif PRESCRIBED_CILIA

  #if USE_BROYDEN_FOR_EVERYTHING
    #error "Prescribed cilia motion does not support using Broyden's method for everything."
  #endif

  #if WRITE_GENERALISED_FORCES
    #undef  PRESCRIBED_BODY_VELOCITIES
    #define PRESCRIBED_BODY_VELOCITIES true
    #undef  DYNAMIC_PHASE_EVOLUTION
    #define DYNAMIC_PHASE_EVOLUTION false
    #undef  DYNAMIC_SHAPE_ROTATION
    #define DYNAMIC_SHAPE_ROTATION false
  #endif

#endif

#if INFINITE_PLANE_WALL
  #undef  NBROY
  #define NBROY (6*NFIL*NSEG)
  #undef  PRESCRIBED_BODY_VELOCITIES
  #define PRESCRIBED_BODY_VELOCITIES true
  #if RPY_MOBILITY
    #undef  RSEG
    #define RSEG 1.0
  #endif
  #undef  MU
  #define MU 1.0
#endif

#if ROD
  #undef  DT
  #define DT (0.1)
#endif

#if NO_CILIA_SQUIRMER
  #if !SURFACE_OF_REVOLUTION_BODIES
    // #error "Squirmer simulations are only compatible with surface-of-revolution bodies."
  #endif
#endif

#if WEAKLY_COUPLED_FILAMENTS_RPY_MOBILITY
  #if !PRESCRIBED_CILIA
    // Only works for filament forces, not torques; avoids barrier_forces() overhead for non-prescribed shapes.
    #error "Weakly-coupled-filaments mobility only supports prescribed-shape filaments."
  #endif
#endif

#if UAMMD_FCM
  #undef  RBLOB
  #define RBLOB RSEG // FCM requires all radii to be equal.
#endif

// FIL_LENGTH is ambiguous by DL; use PRESCRIBED_CILIA definition to match
// Landau-Lifshitz elastic filament results (nothing elastic outside that segment range).
#if PRESCRIBED_CILIA
  #define FIL_LENGTH (DL*(NSEG_PER_CILIA-1))
#else
  #define FIL_LENGTH (DL*NSEG)
#endif

#define DISPLAYTIME false

#endif // MY_CONFIG_HEADER_INCLUDED