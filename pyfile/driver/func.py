import configparser
import os
import util
from filelock import FileLock

class DRIVER:

    def __init__(self):
        self.globals_name = 'input/globals.ini'
        self.afix = ''
        self.inputfile = f""

        self.category = 'tilt_test/'
        self.category = 'resolution/'
        

        self.exe_name = 'cilia_1e-4_newbeat'
        # self.exe_name = 'cilia_1e-4_30_ishikawa'

        # self.date = '20250204_squirmer'
        
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"


        # self.category = 'ic_hpc_sim_free_with_force/'
        # self.exe_name = 'cilia_1e-4_free_with_force'
        # self.date = '20240311_1'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"

        # self.category = 'resolution/'
        # self.date = '20240822_sangani_boxsize2'
        # self.exe_name = 'cilia_1e-6_sangani'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"

        # self.category = 'tilt_test/illustration/'
        # self.exe_name = 'cilia_1e-4_with_force'
        # self.date = '20241029_illustration'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"

        # self.category = 'instability/'
        # self.exe_name = 'cilia_1e-4_instability_double'
        # self.date = '20241028_test'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"


        # self.category = 'ishikawa/'
        # self.exe_name = 'cilia_1e-4_ishikawa_rpy'
        # # self.date = '20240829_pnas_volvox_beat'
        # # self.date = '20240813_pnas_volvox_beat'
        # # self.date = '20240903_real_volvox_slender50'
        # self.date = '20241015_pnas_rpy'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"

        # self.category = 'giant_swimmer/'
        # self.exe_name = 'cilia_1e-4_free_with_force_300'
        # self.date = 'combined_analysis_force_rerun'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"

        # self.category = 'volvox_bicilia/individual_pair/'
        # self.exe_name = 'cilia_1e-4_individual_pair_fixed'
        # self.date = '20241217_fixed_ospread'

        self.category = 'fixed_swimmer_correct/'
        self.exe_name = 'cilia_1e-4_fixed'
        self.date = '20250125_fixed_correct'
        self.dir = f"data/{self.category}{self.date}{self.afix}/"

        self.category = 'resolution/'
        self.exe_name = 'cilia_1e-4_squirmer'
        self.date = '20250214_1e-4_squirmer_fcm'
        self.dir = f"data/{self.category}{self.date}{self.afix}/"

        # self.category = 'regular_wall_sim/'
        # self.exe_name = 'cilia_1e-4_squirmer'
        # self.date = '20250204_1e-4_ref'
        # self.dir = f"data/{self.category}{self.date}{self.afix}/"
        

        self.pars_list = {
                     "index": [],
                     "nswim": [],
                     "nseg": [],
                     "nfil": [],
                     "nblob": [],
                     "ar": [],
                     "spring_factor": [],
                     "tilt_angle": [],
                     "force_mag": [],
                     "seg_sep": [],
                     "period": [],
                     "sim_length": [],
                     "nx": [],
                     "ny": [],
                     "nz": [],
                     "boxsize": [],
                     "fil_spacing": [],
                     "blob_spacing": [],
                     "fil_x_dim": [],
                     "blob_x_dim": [],
                     "hex_num": [],
                     "reverse_fil_direction_ratio": [],
                     "pair_dp": [],
                     "wavnum": [],
                     "wavnum_dia": [],
                     "dimensionless_force": [],
                     "fene_model": [],
                     "force_noise_mag": [],
                     "omega_spread": []}

        self.sweep_shape = (8, 1, 1, 1)
        # self.sweep_shape = (6, 1, 1, 1)

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
        lock = FileLock(f"{self.globals_name}.lock")  # Create a lock file

        with lock:  # Ensure exclusive access
            ini.read(self.globals_name)

            if not ini.has_section(section):
                ini.add_section(section)  # Ensure section exists

            ini.set(section, variable, str(value))

            # Save the changes back to the file
            with open(self.globals_name, 'w') as configfile:
                ini.write(configfile, space_around_delimiters=False)


    def create_rules(self):
        # Define the rule of sweeping simulations
        index = 0
        for i in range(self.sweep_shape[0]):
            for j in range(self.sweep_shape[1]):
                for k in range(self.sweep_shape[2]):
                    for l in range(self.sweep_shape[3]):

                        seg_sep = 2.6
                        force_mag = 1.0
                        tilt_angle = 0.0
                        pair_dp = 1.0
                        wavnum = 0.0
                        wavnum_dia = 0.0
                        period = 1
                        dimensionless_force = 220.
                        fene_model = 0
                        force_noise_mag = 0.0
                        omega_spread = 0.0

                        fil_spacing=80.0
                        blob_spacing=8.0
                        fil_x_dim=20
                        blob_x_dim=200
                        hex_num=2
                        reverse_fil_direction_ratio=0.0

                        # # planar triangle
                        nfil = int(1)
                        nblob = int(400)
                        nseg = 20
                        ar = round(1, 2)
                        period = 1
                        spring_factor = round(0.05, 3)
                        nx=int(32)
                        ny=int(128)
                        nz=int(96)
                        boxsize=20
                        fil_spacing=20.0
                        blob_spacing=2.0
                        fil_x_dim=1
                        blob_x_dim=10
                        hex_num=2
                        reverse_fil_direction_ratio=0.0
                        sim_length = 1
                        force_noise_mag = 0.0
                        omega_spread = 0.0
                        pair_dp = 1.0
                        fene_model = 0

                        # # # callibration
                        # nfil = int(1)
                        # nblob = int(0)
                        # nseg = 20
                        # fil_spacing=256.0
                        # blob_spacing=4.0
                        # fil_x_dim=2
                        # blob_x_dim=64
                        # sim_length = 1
                        # pair_dp = 1.0

                        # # # IVP sim
                        # nfil = 159
                        # nblob = 9000
                        # ar = 8.0
                        
                        # nseg = 20
                        # nx=256
                        # ny=256
                        # nz=256
                        # boxsize=4000
                        # spring_factor = round(0.005, 3)
                        # period = 1
                        # sim_length = 300
                        # tilt_angle = 0.
                        # pair_dp = 1.0
                        # wavnum = 0.0
                        # wavnum_dia = 0.0
                        # fene_model = 1
                        # omega_spread = 0.0
                        # force_noise_mag = 0.0
                        # pair_dp = 1.0
                        

                        # # ishikawa pnas
                        # nfil = [160, 640, 2560][i]
                        # nblob = 40962
                        # ar = 20
                        # nseg = 40
                        # nx=400
                        # ny=400
                        # nz=400
                        # boxsize=8000
                        # spring_factor = round(0)
                        # period = 1
                        # sim_length = 1
                        # tilt_angle = 0.
                        # wavnum = 0.0
                        # wavnum_dia = 0.0
                        # pair_dp = 0.0

                        # ishikawa jfm
                        # nfil = 160
                        # nblob = 40962
                        # ar = 6
                        # nseg = 40
                        # nx=400
                        # ny=400
                        # nz=400
                        # boxsize=8000
                        # spring_factor = [-1, 0, 0.5, 1, 1.5, 2][i]
                        # period = 1
                        # sim_length = 1
                        # tilt_angle = 0

                        # ishikawa resolution
                        # nfil = 160
                        # nblob = int(20 + (3*i)**3)
                        # ar = 6
                        # nseg = 40

                        # nfil = 640
                        # nblob = int(20 + (3*i)**3)
                        # ar = 20
                        # nseg = 40
                        # nx=400
                        # ny=400
                        # nz=400
                        # boxsize=8000
                        # spring_factor = round(0)
                        # period = 1
                        # sim_length = 0.0034
                        # tilt_angle = 0

                        # pair phase difference
                        # nfil = 1278
                        # nblob = 40961
                        # ar = 15.0
                        
                        # nseg = 20
                        # nx=256
                        # ny=256
                        # nz=256
                        # boxsize=4000
                        # spring_factor = round(0.01, 3)
                        # period = 1
                        # sim_length = 300
                        # tilt_angle = 0.
                        # pair_dp = 1.0
                        # wavnum = 0.0
                        # wavnum_dia = 0.0
                        # fene_model = 1
                        # omega_spread = 0.2*i
                        # force_noise_mag = 0000.0*i
                        # pair_dp = 1.0

                        # swimmer size trend
                        # nfil = [159, 639, 1128, 1763, 2539, 4291][i]
                        # nblob = [9000, 40961, 72817, 113777, 163839, 276888][i]
                        # ar = [8.0, 15.0, 20.0, 25.0, 30.0, 39.0][i]
                        # nseg = 20
                        # nx=512
                        # ny=512
                        # nz=512
                        # boxsize=8000
                        # spring_factor = round(0.005, 3)
                        # period = 1
                        # sim_length = 2
                        # tilt_angle = 0

                        # sangani resolution
                        nfil = int(0)
                        nblob = int(9000*(i+1))
                        nseg = 20
                        # ar = round(8*(j+1), 2)
                        ar = round(0.26273*(nblob/4./3.141592653)**.5, 2)
                        period = 1
                        spring_factor = round(0.05, 3)
                        nx=int(400)
                        ny=int(400)
                        nz=int(400)
                        boxsize=12000
                        fil_spacing=20.0
                        blob_spacing=2.0
                        fil_x_dim=1
                        blob_x_dim=10
                        hex_num=2
                        reverse_fil_direction_ratio=0.0
                        sim_length = 0.003
                        force_noise_mag = 0.0
                        omega_spread = 0.0
                        pair_dp = 1.0
                        fene_model = 0

                        self.pars_list["index"].append(index)
                        self.pars_list["nswim"].append(1)
                        self.pars_list["nseg"].append(nseg)
                        self.pars_list["nfil"].append(nfil)
                        self.pars_list["nblob"].append(nblob)
                        self.pars_list["ar"].append(ar)
                        self.pars_list["spring_factor"].append(spring_factor)
                        self.pars_list["force_mag"].append(force_mag)
                        self.pars_list["seg_sep"].append(seg_sep)
                        self.pars_list["period"].append(period)
                        self.pars_list["sim_length"].append(sim_length)
                        self.pars_list["tilt_angle"].append(tilt_angle)
                        self.pars_list["nx"].append(nx)
                        self.pars_list["ny"].append(ny)
                        self.pars_list["nz"].append(nz)
                        self.pars_list["boxsize"].append(boxsize)
                        self.pars_list["fil_spacing"].append(fil_spacing)
                        self.pars_list["blob_spacing"].append(blob_spacing)
                        self.pars_list["fil_x_dim"].append(fil_x_dim)
                        self.pars_list["blob_x_dim"].append(blob_x_dim)
                        self.pars_list["hex_num"].append(hex_num)
                        self.pars_list["reverse_fil_direction_ratio"].append(reverse_fil_direction_ratio)
                        self.pars_list["pair_dp"].append(pair_dp)
                        self.pars_list["wavnum"].append(wavnum)
                        self.pars_list["wavnum_dia"].append(wavnum_dia)
                        self.pars_list["dimensionless_force"].append(dimensionless_force)
                        self.pars_list["fene_model"].append(fene_model)
                        self.pars_list["force_noise_mag"].append(force_noise_mag)
                        self.pars_list["omega_spread"].append(omega_spread)


                        index += 1
        # Write rules to sim list file
        self.write_rules()

    def delete_files(self):
        util.delete_files_in_directory(self.dir)

    def view_files(self):
        util.view_files_in_directory(self.dir)
        print(f"\033[32m{self.dir}\033[m")
        print(f"\033[34m{self.exe_name}\033[m")

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
        except:
            print("WARNING: " + self.dir + "rules.ini not found.")

    def run(self):
        self.create_ini()
        self.write_ini("Filenames", "simulation_dir", self.dir)

        # Read rules from the sim list file
        self.read_rules()

        thread_list = util.even_list_index(self.num_sim, self.num_thread)
        sim_index_start = thread_list[self.current_thread]
        sim_index_end = thread_list[self.current_thread+1]

        print(f"Partitioning {self.num_sim} into {self.num_thread} threads\n" +\
              f"Partition index: {self.current_thread} / {self.num_thread-1} \n" + \
              f"[{sim_index_start} - {sim_index_end}] / {thread_list}\n" +\
              f"on GPU: {self.cuda_device}")
        
        # Iterate through the sim list and write to .ini file and execute
        for i in range(sim_index_start, sim_index_end):
            
            for key, value in self.pars_list.items():
                self.write_ini("Parameters", key, float(self.pars_list[key][i]))
            self.simName = f"ciliate_{self.pars_list['nfil'][i]:.0f}fil_{self.pars_list['nblob'][i]:.0f}blob_{self.pars_list['ar'][i]:.2f}R_{self.pars_list['spring_factor'][i]:.4f}torsion_{self.pars_list['tilt_angle'][i]:.4f}tilt_{self.pars_list['pair_dp'][i]:.4f}dp_{self.pars_list['force_noise_mag'][i]:.4f}noise_{self.pars_list['omega_spread'][i]:.4f}ospread"
            self.write_ini("Filenames", "simulation_file", self.simName)
            self.write_ini("Filenames", "simulation_dir", self.dir)
            self.write_ini("Filenames", "filplacement_file_name", f"input/placement/icosahedron/icosa_d2_N160.dat")
            # self.write_ini("Filenames", "filplacement_file_name", f"input/placement/icosahedron/icosa_d3_N640.dat")
            # self.write_ini("Filenames", "filplacement_file_name", f"input/placement/icosahedron/icosa_d4_N2560.dat")
            self.write_ini("Filenames", "blobplacement_file_name", f"input/placement/icosahedron/icosa_d6_N40962.dat")
            # self.write_ini("Filenames", "blobplacement_file_name", f"input/placement/icosahedron/icosa_d4_N2562.dat")
            self.write_ini("Filenames", "simulation_icstate_name", f"{self.dir}psi{i}.dat")
            self.write_ini("Filenames", "simulation_bodystate_name", f"{self.dir}bodystate{i}.dat")
            self.write_ini("Filenames", "cufcm_config_file_name", f"input/simulation_info_cilia")


            

            # command = f"export OPENBLAS_NUM_THREADS=1; \
            #             export CUDA_VISIBLE_DEVICES={self.cuda_device}; \
            #             ./bin/{self.exe_name} > terminal_outputs/output_{self.date}_{self.pars_list['nfil'][i]:.0f}fil_{i}.out"
            
            
            command = f"export OPENBLAS_NUM_THREADS=1; \
                        export CUDA_VISIBLE_DEVICES={self.cuda_device}; \
                        ./bin/{self.exe_name} "
            
            # override on ic hpc
            if self.run_on_hpc:
                print("\n Running on HPC \n\n\n")
                command = f"export OPENBLAS_NUM_THREADS=1; \
                            ./bin/{self.exe_name}"


            os.system(command)