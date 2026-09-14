import configparser
import os
import numpy as np
import util
from filelock import FileLock

# ---------------------------------------------------------------------------
# Simulation presets
# ---------------------------------------------------------------------------
# Each preset is a dict with:
#   category, date, exe_name       : identify output directory and binary
#   sweep_shape (n0,n1,n2,n3)      : loop bounds for indices i,j,k,l
#   filplacement_file              : icosahedron filament placement file
#   blobplacement_file             : icosahedron blob placement file
#   params                         : parameter name → scalar value or
#                                    callable(i,j,k,l) for swept parameters
# ---------------------------------------------------------------------------

PRESETS = {

    'two_fil': {
        'category':   'for_paper/twofil/',
        'date':       '20250716',
        'exe_name':   'cilia_1e-4_twofil',
        'sweep_shape': (40, 60, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         2,
            'nblob':        0,
            'nseg':         20,
            'ar':           1.0,
            'spring_factor': 0.005,
            'period':       1,
            'sim_length':   100,
            'nx':           128, 'ny': 128, 'nz': 128,
            'boxsize':      400,
            'blob_spacing': 5.0,
            'fil_x_dim':    1,
            'blob_x_dim':   10,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'fil_spacing':  lambda i,j,k,l: ((12.0*i)**2 + (12.0*j)**2)**0.5,
            'fil_x_spacing': 0.0,
            'twofil_angle': lambda i,j,k,l: np.arctan2(12.0*j, 12.0*i),
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'multifil_1d': {
        'category':   'for_paper/multifil/',
        'date':       '20250802',
        'exe_name':   'cilia_1e-4_plane',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         45,
            'nblob':        0,
            'nseg':         20,
            'ar':           1.0,
            'spring_factor': lambda i,j,k,l: round(0.0 + 0.001*i, 3),
            'period':       1,
            'sim_length':   500,
            'nx':           128, 'ny': 128, 'nz': 128,
            'boxsize':      400,
            'fil_spacing':  49.4,
            'fil_x_spacing': 0.0,
            'blob_spacing': 50.0,
            'fil_x_dim':    1,
            'blob_x_dim':   10,
            'hex_num':      1,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': np.pi/2,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'multifil_2d': {
        'category':   'for_paper/multifil/',
        'date':       '20250802',
        'exe_name':   'cilia_1e-4_plane',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         lambda i,j,k,l: int(45 + 45*i),
            'nblob':        0,
            'nseg':         20,
            'ar':           1.0,
            'spring_factor': 0.005,
            'period':       1,
            'sim_length':   500,
            'nx':           128, 'ny': 128, 'nz': 128,
            'boxsize':      400,
            'fil_spacing':  49.4,
            'fil_x_spacing': lambda i,j,k,l: 49.4/2*3**0.5,
            'blob_spacing': 50.0,
            'fil_x_dim':    lambda i,j,k,l: 1 + i,
            'blob_x_dim':   20,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': np.pi/2,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'calibration': {
        'category':   'regular_wall_sim/',
        'date':       '20260831_temp_forcing',
        'exe_name':   'cilia_1e-4_calibration',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         1,
            'nblob':        0,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': 0.005,
            'period':       1,
            'sim_length':   1,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      4000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.0,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'ivp_sim': {
        'category':   'ic_hpc_sim_free_with_force2/',
        'date':       '20240311_1',
        'exe_name':   'cilia_1e-4_free',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         639,
            'nblob':        40961,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': lambda i,j,k,l: round(0.1 + 0.1*i, 3),
            'period':       1,
            'sim_length':   1,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      4000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.0,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'ishikawa_pnas': {
        'category':   'ishikawa/',
        'date':       '20241015_pnas_rpy',
        'exe_name':   'cilia_1e-4_ishikawa_rpy',
        'sweep_shape': (3, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         lambda i,j,k,l: [160, 640, 2560][i],
            'nblob':        40962,
            'nseg':         40,
            'ar':           20,
            'spring_factor': 0,
            'period':       1,
            'sim_length':   1,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.0,
            'wavnum_dia':   0.0,
            'pair_dp':      0.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'ishikawa_jfm': {
        'category':   'ishikawa/',
        'date':       '20241015_pnas_rpy',
        'exe_name':   'cilia_1e-4_ishikawa_rpy',
        'sweep_shape': (6, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d2_N160.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         160,
            'nblob':        40962,
            'nseg':         40,
            'ar':           6,
            'spring_factor': lambda i,j,k,l: [-1, 0, 0.5, 1, 1.5, 2][i],
            'period':       1,
            'sim_length':   1,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'ishikawa_resolution': {
        'category':   'resolution/',
        'date':       '20240822_sangani_boxsize2',
        'exe_name':   'cilia_1e-6_sangani',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         640,
            'nblob':        lambda i,j,k,l: int(20 + (3*i)**3),
            'nseg':         40,
            'ar':           20,
            'spring_factor': 0,
            'period':       1,
            'sim_length':   0.0034,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'bicilia_ishikawa': {
        'category':   'volvox/',
        'date':       '20260319_dp_sweep',
        'exe_name':   'cilia_1e-4_bicilia_ishikawa2',
        'sweep_shape': (10, 4, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         640,
            'nblob':        40962,
            'nseg':         40,
            'ar':           15.0,
            'spring_factor': 0.005,
            'period':       1,
            'sim_length':   1,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       lambda i,j,k,l: [-2.35, -1, 0, 1][j],
            'wavnum_dia':   0.0,
            'pair_dp':      lambda i,j,k,l: round(0.1 * i, 2),
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'pair_phase_diff': {
        'category':   'volvox_bicilia/individual_pair/',
        'date':       '20241217_fixed_ospread',
        'exe_name':   'cilia_1e-4_individual_pair_fixed',
        'sweep_shape': (10, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         639,
            'nblob':        40961,
            'nseg':         40,
            'ar':           15.0,
            'spring_factor': 0.01,
            'period':       1,
            'sim_length':   200,
            'nx':           440, 'ny': 440, 'nz': 440,
            'boxsize':      4000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       1,
            'wavnum_dia':   0.0,
            'pair_dp':      lambda i,j,k,l: round(1.0 - 0.1*i, 2),
            'fene_model':   1,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'swimmer_size_trend': {
        'category':   'giant_swimmer/',
        'date':       'combined_analysis_force_rerun',
        'exe_name':   'cilia_1e-4_free',
        'sweep_shape': (6, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         lambda i,j,k,l: [159, 639, 1128, 1763, 2539, 4291][i],
            'nblob':        lambda i,j,k,l: [9000, 40961, 72817, 113777, 163839, 276888][i],
            'nseg':         20,
            'ar':           lambda i,j,k,l: [8.0, 15.0, 20.0, 25.0, 30.0, 39.0][i],
            'spring_factor': 0.005,
            'period':       1,
            'sim_length':   2,
            'nx':           512, 'ny': 512, 'nz': 512,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'sangani_resolution': {
        'category':   'resolution/',
        'date':       '20240822_sangani_boxsize2',
        'exe_name':   'cilia_1e-6_sangani',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         0,
            'nblob':        lambda i,j,k,l: int(8000*(i+1)),
            'nseg':         20,
            'ar':           lambda i,j,k,l: round(0.26273*(8000*(i+1)/4./np.pi)**0.5, 2),
            'spring_factor': 0.05,
            'period':       1,
            'sim_length':   0.003,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      12000,
            'fil_spacing':  20.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 2.0,
            'fil_x_dim':    1,
            'blob_x_dim':   10,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'volvox_newbeat': {
        'category':   'volvox/',
        'date':       '20260902_newbeat',
        'exe_name':   'cilia_1e-4_free_newbeat',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         639,
            'nblob':        40961,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': 0.1,
            'period':       1,
            'sim_length':   15,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      4000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'volvox_oldbeat': {
        'category':   'volvox/',
        'date':       '20260902_oldbeat',
        'exe_name':   'cilia_1e-4_free_oldbeat',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         639,
            'nblob':        40961,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': 0.1,
            'period':       1,
            'sim_length':   15,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      4000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.5,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   0,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'pair_freq_sweep': {
        'category':   'pair/',
        'date':       '20260909_pair_freq',
        'exe_name':   'cilia_1e-4_pair',
        'sweep_shape': (20, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         2,
            'nblob':        1,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': 0.1,
            'period':       1,
            'sim_length':   300,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.0,
            'wavnum_dia':   0.0,
            'pair_dp':      lambda i,j,k,l: round(1 - 0.05 * i, 3),
            'fene_model':   1,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'rbf2_calibration': {
        'category':   'volvox/',
        'date':       '20260914_rbf2_calib',
        'exe_name':   'cilia_1e-4_pair_calib',
        'sweep_shape': (1, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         1,
            'nblob':        0,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': 0.1,
            'period':       1,
            'sim_length':   5,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    1,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.0,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   1,
            'force_noise_mag': 0.0,
            'phase_noise_mag': 0.0,
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

    'pair_noise_sweep': {
        'category':   'pair/',
        'date':       '20260910_pair_noise',
        'exe_name':   'cilia_1e-4_pair',
        'sweep_shape': (20, 1, 1, 1),
        'filplacement_file':  'input/placement/icosahedron/icosa_d3_N640.dat',
        'blobplacement_file': 'input/placement/icosahedron/icosa_d6_N40962.dat',
        'params': {
            'nfil':         2,
            'nblob':        1,
            'nseg':         20,
            'ar':           15.0,
            'spring_factor': 0.1,
            'period':       1,
            'sim_length':   300,
            'nx':           400, 'ny': 400, 'nz': 400,
            'boxsize':      8000,
            'fil_spacing':  80.0,
            'fil_x_spacing': 0.0,
            'blob_spacing': 8.0,
            'fil_x_dim':    20,
            'blob_x_dim':   200,
            'hex_num':      2,
            'reverse_fil_direction_ratio': 0.0,
            'twofil_angle': 0.0,
            'tilt_angle':   0.0,
            'force_mag':    1.0,
            'seg_sep':      2.6,
            'wavnum':       0.0,
            'wavnum_dia':   0.0,
            'pair_dp':      1.0,
            'fene_model':   1,
            'force_noise_mag': 0.0,
            # onset at σ_ψ ~ 10; 20-point sweep with fine spacing through the transition
            'phase_noise_mag': lambda i,j,k,l: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                                  8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0][i],
            'omega_spread': 0.0,
            'dimensionless_force': 220.0,
        },
    },

}

ACTIVE_PRESET = 'rbf2_calibration'

# ---------------------------------------------------------------------------

class DRIVER:

    def __init__(self):
        self.globals_name = 'input/globals.ini'
        self.afix = ''
        self.inputfile = f""

        preset = PRESETS[ACTIVE_PRESET]
        self.category = preset['category']
        self.date     = preset['date']
        self.exe_name = preset['exe_name']
        self.dir      = f"data/{self.category}{self.date}{self.afix}/"
        self.sweep_shape = preset['sweep_shape']

        self.pars_list = {
            "index": [], "nswim": [], "nseg": [], "nfil": [], "nblob": [],
            "ar": [], "spring_factor": [], "tilt_angle": [], "force_mag": [],
            "seg_sep": [], "period": [], "sim_length": [], "nx": [], "ny": [],
            "nz": [], "boxsize": [], "fil_spacing": [], "fil_x_spacing": [],
            "blob_spacing": [], "fil_x_dim": [], "blob_x_dim": [], "hex_num": [],
            "reverse_fil_direction_ratio": [], "twofil_angle": [], "pair_dp": [],
            "wavnum": [], "wavnum_dia": [], "dimensionless_force": [],
            "fene_model": [], "force_noise_mag": [], "phase_noise_mag": [], "omega_spread": [],
        }

        self.num_sim = 0
        self.current_thread = 0
        self.num_thread = 1
        self.cuda_device = 0
        self.run_on_hpc = False

    def update_date(self, date):
        self.date = date
        self.dir = f"data/{self.category}{self.date}{self.afix}/"

    def create_ini(self):
        ini = configparser.ConfigParser()
        ini.add_section('Parameters')
        ini.add_section('Filenames')
        ini.add_section('Box')
        ini.add_section('Hex')
        ini.add_section('Concentric')
        ini.add_section('Seeding_util')
        with open(self.globals_name, 'w') as configfile:
            ini.write(configfile, space_around_delimiters=False)

    def write_ini(self, section, variable, value):
        ini = configparser.ConfigParser()
        lock = FileLock(f"{self.globals_name}.lock")

        with lock:
            ini.read(self.globals_name)

            if not ini.has_section(section):
                ini.add_section(section)

            ini.set(section, variable, str(value))

            with open(self.globals_name, 'w') as configfile:
                ini.write(configfile, space_around_delimiters=False)

    def create_rules(self):
        preset = PRESETS[ACTIVE_PRESET]
        params = preset['params']
        param_keys = [k for k in self.pars_list if k not in ('index', 'nswim')]

        index = 0
        for i in range(self.sweep_shape[0]):
            for j in range(self.sweep_shape[1]):
                for k in range(self.sweep_shape[2]):
                    for l in range(self.sweep_shape[3]):
                        p = {key: (val(i, j, k, l) if callable(val) else val)
                             for key, val in params.items()}

                        self.pars_list["index"].append(index)
                        self.pars_list["nswim"].append(1)
                        for key in param_keys:
                            self.pars_list[key].append(p[key])

                        index += 1

        self.write_rules()

    def delete_files(self):
        util.delete_files_in_directory(self.dir)

    def view_files(self):
        util.view_files_in_directory(self.dir)
        print(f"\033[1;33mPreset : {ACTIVE_PRESET}\033[m")
        print(f"\033[32m{self.dir}\033[m")
        print(f"\033[34m{self.exe_name}\033[m")
        # Show swept parameters (those with a callable value in the preset)
        preset_params = PRESETS[ACTIVE_PRESET]['params']
        sweep_shape   = PRESETS[ACTIVE_PRESET]['sweep_shape']
        swept = {k: v for k, v in preset_params.items() if callable(v)}
        if swept:
            indices = range(sweep_shape[0])
            header  = "  {:>6}  ".format("sim") + "  ".join(f"{k:>18}" for k in swept)
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i in indices:
                row = f"  {i:>6}  " + "  ".join(
                    f"{v(i,0,0,0):>18.4g}" for v in swept.values()
                )
                print(row)

    def check_rules(self):
        from pathlib import Path
        file_path = Path(self.dir + 'rules.ini')
        if file_path.is_file():
            print("Using the existing rules.ini in the directory\n\n\n")
        else:
            print("rules.ini does not exist. Applying new rules.\n\n\n")
        return file_path.is_file()

    def write_rules(self):
        os.system(f'mkdir -p {self.dir}')
        sim = configparser.ConfigParser()
        sim.add_section('Parameter list')
        for key, value in self.pars_list.items():
            sim['Parameter list'][key] = ', '.join(map(str, value))
        with open(self.dir+"rules.ini", 'w') as configfile:
            sim.write(configfile, space_around_delimiters=False)

    def read_rules(self):
        sim = configparser.ConfigParser()
        try:
            sim.read(self.dir+"rules.ini")
            for key, value in self.pars_list.items():
                if(key in sim["Parameter list"]):
                    self.pars_list[key] = [float(x) for x in sim["Parameter list"][key].split(', ')][0::1]
            self.num_sim = len(self.pars_list["nfil"])
            if len(self.pars_list['fil_x_spacing']) == 0:
                self.pars_list['fil_x_spacing'] = [0.0] * self.num_sim
        except:
            print("WARNING: " + self.dir + "rules.ini not found.")

    def run(self):
        self.create_ini()
        self.write_ini("Filenames", "simulation_dir", self.dir)

        self.read_rules()

        thread_list = util.even_list_index(self.num_sim, self.num_thread)
        sim_index_start = thread_list[self.current_thread]
        sim_index_end = thread_list[self.current_thread+1]

        print(f"Partitioning {self.num_sim} into {self.num_thread} threads\n" +
              f"Partition index: {self.current_thread} / {self.num_thread-1} \n" +
              f"[{sim_index_start} - {sim_index_end}] / {thread_list}\n" +
              f"on GPU: {self.cuda_device}")

        preset = PRESETS[ACTIVE_PRESET]

        for i in range(sim_index_start, sim_index_end):

            for key, value in self.pars_list.items():
                self.write_ini("Parameters", key, float(self.pars_list[key][i]))

            self.simName = (
                f"ciliate_{self.pars_list['nfil'][i]:.0f}fil"
                f"_{self.pars_list['nblob'][i]:.0f}blob"
                f"_{self.pars_list['ar'][i]:.2f}R"
                f"_{self.pars_list['spring_factor'][i]:.4f}torsion"
                f"_{self.pars_list['tilt_angle'][i]:.4f}tilt"
                f"_{self.pars_list['pair_dp'][i]:.4f}dp"
                f"_{self.pars_list['force_noise_mag'][i]:.4f}noise"
                f"_{self.pars_list['omega_spread'][i]:.4f}ospread"
                f"_{self.pars_list['index'][i]:.0f}index"
            )
            self.write_ini("Filenames", "simulation_file", self.simName)
            self.write_ini("Filenames", "simulation_dir", self.dir)
            self.write_ini("Filenames", "filplacement_file_name",  preset['filplacement_file'])
            self.write_ini("Filenames", "blobplacement_file_name", preset['blobplacement_file'])
            self.write_ini("Filenames", "simulation_icstate_name",   f"{self.dir}psi.dat")
            self.write_ini("Filenames", "simulation_bodystate_name", f"{self.dir}bodystate{i}.dat")
            self.write_ini("Filenames", "cufcm_config_file_name",    f"input/simulation_info_cilia")

            command = (f"export OPENBLAS_NUM_THREADS=1; "
                       f"export CUDA_VISIBLE_DEVICES={self.cuda_device}; "
                       f"./bin/{self.exe_name} ")

            if self.run_on_hpc:
                print("\n Running on HPC \n\n\n")
                command = (f"export OPENBLAS_NUM_THREADS=1; "
                           f"./bin/{self.exe_name}")

            os.system(command)
