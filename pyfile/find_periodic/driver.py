import configparser
import subprocess
import os
import math
import numpy as np

class DRIVER:

    def __init__(self):
        self.globals_name = 'input/globals.ini'
        self.exe_name = 'cilia_double_1e-8'
        self.category = 'JFNK/'

        self.date = '20240320_JFNK_d'
        self.afix = ''
        # self.dir = f"data/expr_sims/{self.date}{self.afix}/"
        self.dir = f"data/{self.category}{self.date}{self.afix}/"

        self.pars_list = {
                     "nswim": [],
                     "nseg": [],
                     "nfil": [],
                     "nblob": [],
                     "ar": [],
                     "spring_factor": [],
                     "force_mag": [],
                     "seg_sep": [],
                     "period": [],
                     "sim_length": []}
    
        self.ite = 0
        
        self.current_thread = 0
        self.num_thread = 1
        self.cuda_device = 5
    
    def create_ini(self):
        ini = configparser.ConfigParser()
        ini.add_section('Parameters')
        ini.add_section('Filenames')
        with open(self.globals_name, 'w') as configfile:
            ini.write(configfile, space_around_delimiters=False)
        

    def write_ini(self, section, variable, value):
        ini = configparser.ConfigParser()
        ini.read(self.globals_name)

        ini.set(section, variable, f'{value}')

        # Save the changes back to the file
        with open(self.globals_name, 'w') as configfile:
            ini.write(configfile, space_around_delimiters=False)


    def change_variables(self, nfil, nseg, nblob, ar, k, period, sim_length):
        nseg = nseg
        nfil = nfil
        nblob = nblob
        ar = ar
        sim_length = sim_length
        force_mag = 1
        seg_sep = 2.6
        
        self.pars_list["nswim"].append(1)
        self.pars_list["nseg"].append(nseg)
        self.pars_list["nfil"].append(nfil)
        self.pars_list["nblob"].append(nblob)
        self.pars_list["ar"].append(ar)
        self.pars_list["spring_factor"].append(k)
        self.pars_list["force_mag"].append(force_mag)
        self.pars_list["seg_sep"].append(seg_sep)
        self.pars_list["period"].append(period)
        self.pars_list["sim_length"].append(sim_length)


    def update_globals_file(self):
        self.create_ini()

        # ite = len(self.pars_list['period'])
        readphase_index = ''
        # Iterate through the sim list and write to .ini file and execute
        for key, value in self.pars_list.items():
            self.write_ini("Parameters", key, float(self.pars_list[key][-1]))
        self.simName = f"ciliate_{self.pars_list['nfil'][-1]:.0f}fil_{self.pars_list['nblob'][-1]:.0f}blob_{self.pars_list['ar'][-1]:.2f}R_{self.pars_list['spring_factor'][-1]:.4f}torsion"
        self.write_ini("Filenames", "simulation_dir", self.dir)
        self.write_ini("Filenames", "simulation_file", self.simName)
        self.write_ini("Filenames", "simulation_icstate_name", self.dir + f"psi.dat")

        return True

    def run(self):

        k = float(self.pars_list["spring_factor"][0])
        command = f"export OPENBLAS_NUM_THREADS=1; \
                    export CUDA_VISIBLE_DEVICES={self.cuda_device}; \
                    ./bin/{self.exe_name} > terminal_outputs/output_{self.date}_{self.pars_list['nfil'][0]:.0f}fil_{k}.out"
        # command = f"export OPENBLAS_NUM_THREADS=1; \
        #             export CUDA_VISIBLE_DEVICES={self.cuda_device}; \
        #             ./bin/{self.exe_name}"

        
        os.system(command)
        self.ite += 1