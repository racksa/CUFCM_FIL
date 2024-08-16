import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import myIo
import util
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.animation as animation
import configparser
from sklearn.cluster import KMeans
import time
import matplotlib as mpl
import os
from scipy.optimize import curve_fit
from numba import cuda, float64

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})


class VISUAL:

    def __init__(self):
        self.globals_name = 'globals.ini'


        self.date = '20240608'
        # self.dir = f"data/IVP159/{self.date}/"

        self.date = '20240717_rpy_get_drag'
        self.dir = f"data/regular_wall_sim/{self.date}/"

        self.date = '20240724_symplectic'
        self.dir = f"data/tilt_test/makeup_pattern/{self.date}/"

        # self.date = '20240730_newbeat'
        # self.dir = f"data/tilt_test/{self.date}/"

        

        # self.date = '20240311_8'
        # self.dir = f"data/ic_hpc_sim/{self.date}/"
        

        self.date = '20240311_2'
        self.dir = f"data/ic_hpc_sim_free/{self.date}/"

        self.date = 'combined_analysis'
        self.dir = f"data/giant_swimmer/{self.date}/"

        # self.date = '20240311_1'
        # self.dir = f"data/ic_hpc_sim_free_with_force/{self.date}/"        

        # self.date = '20240731_pnas_L1'
        # self.date = '20240813_pnas_volvox_beat'
        # self.date = '20240802_pnas_original_beat'
        # self.date = '20240807_ishikawa_resolution6'
        # self.dir = f"data/ishikawa/{self.date}/"

        # self.date = '20240115_resolution'
        # self.dir = f"data/resolution/{self.date}/"



        # self.date = f'index1_alpha0.16326530612244897'
        # self.dir = f"data/bisection/k0.020/section6/iteration2_1e-7/{self.date}/"
        
        # self.date = 'index6_alpha0.875'
        # self.dir = f"data/ic_hpc_bisection/k0.005/iteration1/{self.date}/"


        # self.date = '20240507'
        # self.dir = f"data/regular_wall_sim/{self.date}/"

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
                     "reverse_fil_direction_ratio": []}
        self.video = False
        self.interpolate = False
        self.angle = False
        self.output_to_fcm = False
        self.output_to_superpunto = True
        self.periodic = False
        

        self.show_poles = True
        self.big_sphere = True
        self.noblob = False

        self.planar = False

        if(self.planar):
            self.big_sphere = False
            self.show_poles = False

        self.check_overlap = False


        self.plot_end_frame_setting = 150000
        self.frames_setting = 60

        self.plot_end_frame = self.plot_end_frame_setting
        self.frames = self.frames_setting

        self.select_elst = 0

        self.Lx = 1000
        self.Ly = 1000
        self.Lz = 1000

        self.ncol = 12
        self.num_sim = 0

        self.plot_interval = 3
        
        self.index = 0

    def write_data(self, x, r, filename, box=True, center=True, superpunto=True, color=0):
        plot_x = x.copy()
        if(box):
            plot_x[0] = util.box(plot_x[0], self.Lx)
            plot_x[1] = util.box(plot_x[1], self.Ly)
            plot_x[2] = util.box(plot_x[2], self.Lz)
            if(center):
                plot_x[0] -= 0.5*self.Lx
                plot_x[1] -= 0.5*self.Ly
        if superpunto:
            plot_x[1] = -plot_x[1]
            plot_x[2] = -plot_x[2]
            myIo.write_line(str(plot_x[0]) + ' ' +\
                            str(plot_x[1]) + ' ' +\
                            str(plot_x[2]) + ' ' +\
                            str(r) + ' ' +\
                            str(color),
                            filename)
        else:
            myIo.write_line(str(plot_x[0]) + ' ' +\
                            str(plot_x[1]) + ' ' +\
                            str(plot_x[2]),
                            filename)
        
    def read_rules(self):
        sim = configparser.ConfigParser()
        try:
            sim.read(self.dir+"rules.ini")
            num_fil = len(np.unique([float(s) for s in sim["Parameter list"]['nfil'].split(', ')]))
            num_ar = len(np.unique([float(s) for s in sim["Parameter list"]['ar'].split(', ')]))
            num_elst = len(np.unique([float(s) for s in sim["Parameter list"]['spring_factor'].split(', ')]))
            num_per_elst = int(num_fil*num_ar)
            select_elst = min(num_elst-1, self.select_elst)
            for key, value in self.pars_list.items():
                if(key in sim["Parameter list"]):
                    self.pars_list[key] = [float(x) for x in sim["Parameter list"][key].split(', ')]
            self.num_sim = len(self.pars_list["nfil"])
            if(len(self.pars_list['tilt_angle'])==0):
                self.pars_list['tilt_angle'] = np.zeros(np.shape(self.pars_list['nfil']))
           
        except:
            print("WARNING: " + self.dir + "rules.ini not found.")

    def select_sim(self):

        if(self.index>len(self.pars_list['nfil'])):
            self.index = len(self.pars_list['nfil'])-1
            print(f'Index out of range. Using the last sim: {self.index}')
        self.nseg = int(self.pars_list['nseg'][self.index])
        self.nswim = 1
        self.nfil = int(self.pars_list['nfil'][self.index])
        self.nblob = int(self.pars_list['nblob'][self.index])
        self.ar = self.pars_list['ar'][self.index]
        self.spring_factor = self.pars_list['spring_factor'][self.index]
        self.N = int(self.nswim*(self.nfil*self.nseg + self.nblob))

        
        try:
            self.tilt_angle = self.pars_list['tilt_angle'][self.index]
            self.simName = self.dir + f"ciliate_{self.nfil:.0f}fil_{self.nblob:.0f}blob_{self.ar:.2f}R_{self.spring_factor:.4f}torsion_{self.tilt_angle:.4f}tilt"
            open(self.simName + '_fil_references.dat')
        except:
            self.tilt_angle = 0.
            self.simName = self.dir + f"ciliate_{self.nfil:.0f}fil_{self.nblob:.0f}blob_{self.ar:.2f}R_{self.spring_factor:.4f}torsion"
            try:
                open(self.simName + '_fil_references.dat')
            except:
                self.simName = self.dir + f"ciliate_{self.nfil:.0f}fil_{self.nblob:.0f}blob_{self.ar:.2f}R_{self.spring_factor:.3f}torsion"
            try:
                open(self.simName + '_fil_references.dat')
            except:
                self.simName = self.dir + f"ciliate_{self.nfil:.0f}fil_{self.nblob:.0f}blob_{self.ar:.2f}R_{self.spring_factor:.2f}torsion"
        

        try:
            self.fil_spacing = self.pars_list['fil_spacing'][self.index]
            self.fil_x_dim = self.pars_list['fil_x_dim'][self.index]
        except:
            pass

        
        self.fil_references = myIo.read_fil_references(self.simName + '_fil_references.dat')
        try:
            self.fil_q = myIo.read_fil_references(self.simName + '_fil_q.dat')
        except:
            pass

        self.pars = myIo.read_pars(self.simName + '.par')
        self.radius = 0.5*self.ar*self.pars['FIL_LENGTH']
        self.dt = self.pars['DT']*self.pars['PLOT_FREQUENCY_IN_STEPS']
        self.period = float(self.pars['STEPS_PER_PERIOD'])/float(self.pars['PLOT_FREQUENCY_IN_STEPS'])
        self.fillength = float(self.pars['FIL_LENGTH'])
        self.fildensity = self.nfil*self.fillength**2/(self.radius**2)

        if(not 'PRESCRIBED_CILIA' in self.pars):
            self.pars['PRESCRIBED_CILIA'] = 0
        if(self.pars['NBLOB']>0):
            self.blob_references = myIo.read_blob_references(self.simName + '_blob_references.dat')

        self.plot_end_frame = min(self.plot_end_frame_setting, sum(1 for line in open(self.simName + '_body_states.dat')))
        self.plot_start_frame = max(0, self.plot_end_frame-self.frames_setting)
        self.frames = self.plot_end_frame - self.plot_start_frame

        print(f'index={self.index} file={self.simName}')
        
    def plot(self):
        self.select_sim()
        ax = plt.figure().add_subplot(projection='3d')

        superpuntoDatafileName = f"{self.simName}_superpunto_{self.date}.dat"
        myIo.clean_file(superpuntoDatafileName)

        seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        # if(self.pars['NBLOB']>0):
        #     blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
        # seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
        if (self.pars['PRESCRIBED_CILIA'] == 1):
            fil_states_f = open(self.simName + '_true_states.dat', "r")

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            if(self.check_overlap):
                segs_list = np.zeros((int(self.nfil*self.nseg), 3))
                blobs_list = np.zeros((int(self.nblob), 3))
                

            body_states_str = body_states_f.readline()
            if(self.pars['NFIL']>0):
                seg_states_str = seg_states_f.readline()
                if (self.pars['PRESCRIBED_CILIA'] == 1):
                    fil_states_str = fil_states_f.readline()

                    fil_phases = np.array(fil_states_str.split()[2:2+self.nfil], dtype=float)
                    fil_phases = util.box(fil_phases, 2*np.pi)

            if(i%self.plot_interval==0 and i>=self.plot_start_frame):
                body_states = np.array(body_states_str.split()[1:], dtype=float)
                if(self.pars['NFIL']>0):
                    seg_states = np.array(seg_states_str.split()[1:], dtype=float)                
                
                myIo.write_line('#', superpuntoDatafileName)
                for swim in range(int(self.pars['NSWIM'])):
                    body_pos = body_states[7*swim : 7*swim+3]
                    R = util.rot_mat(body_states[7*swim+3 : 7*swim+7])
                    # R = np.linalg.inv(R)
                    # R = np.eye(3)
                    if(not self.big_sphere or self.check_overlap):
                        for blob in range(int(self.pars['NBLOB'])):
                            blob_x, blob_y, blob_z = util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3])
                            if(self.check_overlap):
                                blobs_list[blob] = blob_x, blob_y, blob_z
                            elif(not self.big_sphere):
                                color=16777215 #white
                                color=13882323 # grey
                                # color=0 # black
                                if not self.noblob:
                                    self.write_data([blob_x, blob_y, blob_z], float(self.pars['RBLOB']), superpuntoDatafileName, self.periodic, color=color)

                    if(self.big_sphere):
                        self.write_data(body_pos, self.radius, superpuntoDatafileName, self.periodic, color=16777215)
                    
                    if(self.show_poles):
                        self.write_data(body_pos + np.matmul(R, np.array([0,0,self.radius])), 3*float(self.pars['RBLOB']), superpuntoDatafileName, self.periodic, color=0*65536+0*256+255)
                        self.write_data(body_pos + np.matmul(R, np.array([0,0,-self.radius])), 3*float(self.pars['RBLOB']), superpuntoDatafileName, self.periodic, color=0*65536+0*256+255)

                    for fil in range(int(self.pars['NFIL'])):
                        fil_color = int("000000", base=16)
                        # Robot arm to find segment position (Ignored plane rotation!)
                        if (self.pars['PRESCRIBED_CILIA'] == 0):
                            fil_i = int(4*fil*self.pars['NSEG'])
                            fil_base_x, fil_base_y, fil_base_z = body_pos + np.matmul(R, self.fil_references[3*fil : 3*fil+3])
                            old_seg_pos = np.array([fil_base_x, fil_base_y, fil_base_z])
                        elif (self.pars['PRESCRIBED_CILIA'] == 1):
                            fil_i = int(3*fil*self.pars['NSEG'])
                            old_seg_pos = seg_states[fil_i : fil_i+3]
                            
                            # WRITE A FUNCTION FOR THIS!!
                            cmap_name = 'hsv'
                            cmap = plt.get_cmap(cmap_name)
                            rgb_color = cmap(fil_phases[fil]/(2*np.pi))[:3]  # Get the RGB color tuple
                            rgb_hex = mcolors.rgb2hex(rgb_color)[1:]  # Convert RGB to BGR hexadecimal format
                            bgr_hex = rgb_hex[4:]+rgb_hex[2:4]+rgb_hex[:2]
                            fil_color = int(bgr_hex, base=16)
                            # fil_color = color=0*65536+0*256+255
                            # print("\n", bgr_hex, fil_color, "\t")
                        if(self.check_overlap):
                            segs_list[fil*self.nseg] = old_seg_pos
                        else:
                            self.write_data(old_seg_pos, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)
                            # self.write_data(old_seg_pos, 10*float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=255*65536+0*256+0)

                        for seg in range(1, int(self.pars['NSEG'])):
                            if (self.pars['PRESCRIBED_CILIA'] == 0):
                                q1 = seg_states[fil_i+4*(seg-1) : fil_i+4*seg]
                                q2 = seg_states[fil_i+4*seg : fil_i+4*seg+4]
                                
                                t1 = util.find_t(q1)
                                t2 = util.find_t(q2)
                                
                                seg_pos = old_seg_pos + 0.5*self.pars['DL']*(t1 + t2)
                                old_seg_pos = seg_pos
                            elif (self.pars['PRESCRIBED_CILIA'] == 1):
                                seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)] 
                            if(self.check_overlap):
                                segs_list[fil*self.nseg + seg] = seg_pos
                            else:
                                self.write_data(seg_pos, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)
                            
                if(self.check_overlap):
                    threshold = 1.0
                    cell_size = 10
                    particle_list = np.concatenate([segs_list, blobs_list])

                    colliding_indices, colliding_particles = util.label_colliding_particles_with_3d_cell_list(particle_list, cell_size, threshold*float(self.pars['RSEG']))
                    
                    self.write_data(body_pos, self.radius, superpuntoDatafileName, self.periodic, color=16777215)
                    print(f'Overlapping case at threshold {threshold} = {len(colliding_indices)}')

                    for i, pos in enumerate(segs_list):
                        fil_color = 16777215
                        if(i in colliding_indices):
                            fil_color = 255
                        self.write_data(pos, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)
                        
    def plot_eco(self):
        self.select_sim()

        cmap_name = 'hsv'
        # cmap_name = 'twilight_shifted'

        # Fourier coeffs for the shape
        Ay = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
                    [4.0318e-01, -1.5553e+00, 7.3455e-01], \
                    [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
                    [8.1046e-02, -3.0982e-01, 1.4568e-01]])

        Ax = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
                    [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
                    [1.6209e-01, -3.4983e-01, 1.9082e-01], \
                    [1.0259e-02, 3.5907e-02, -6.8736e-02]])

        By = np.array([[0, 0, 0], \
                    [2.9136e-01, 1.0721e+00, -1.0433e+00], \
                    [6.1554e-03, 3.2521e-01, -2.8315e-01], \
                    [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

        Bx = np.array([[0, 0, 0], \
                    [1.9697e-01, -5.1193e-01, 3.4778e-01], \
                    [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
                    [1.2311e-02, 1.4157e-01, -1.1695e-01]])
        
        def fitted_shape(s, phase):
            pos = np.zeros(3)
            svec = np.array([s, s**2, s**3])
            fourier_dim = np.shape(Ax)[0]
            cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
            sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])
            cosvec[0] *= 0.5

            x = (cosvec@Ax + sinvec@Bx)@svec
            y = (cosvec@Ay + sinvec@By)@svec
            z = np.zeros(np.shape(x))

            return x, y, z

        superpuntoDatafileName = f"{self.simName}_superpunto_{self.date}.dat"
        myIo.clean_file(superpuntoDatafileName)

        # seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        if (self.pars['PRESCRIBED_CILIA'] == 1):
            fil_states_f = open(self.simName + '_true_states.dat', "r")

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            if(self.check_overlap):
                segs_list = np.zeros((int(self.nfil*self.nseg), 3))
                blobs_list = np.zeros((int(self.nblob), 3))
                
            body_states_str = body_states_f.readline()
            if(self.pars['NFIL']>0):
                # seg_states_str = seg_states_f.readline()
                if (self.pars['PRESCRIBED_CILIA'] == 1):
                    fil_states_str = fil_states_f.readline()

                    fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                    fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)
                    
                    fil_phases = fil_states[:self.nfil]
                    fil_angles = fil_states[self.nfil:]

            if(i%self.plot_interval==0 and i>=self.plot_start_frame):
                body_states = np.array(body_states_str.split()[1:], dtype=float)
                # if(self.pars['NFIL']>0):
                #     seg_states = np.array(seg_states_str.split()[1:], dtype=float)                
                
                myIo.write_line('#', superpuntoDatafileName)
                for swim in range(int(self.pars['NSWIM'])):
                    body_pos = body_states[7*swim : 7*swim+3]
                    R = util.rot_mat(body_states[7*swim+3 : 7*swim+7])

                    if(not self.big_sphere or self.check_overlap):
                        for blob in range(int(self.pars['NBLOB'])):
                            blob_x, blob_y, blob_z = util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3])
                            if(self.check_overlap):
                                blobs_list[blob] = blob_x, blob_y, blob_z
                            elif(not self.big_sphere):
                                color=16777215 #white
                                color=13882323 # grey
                                # color=0 # black
                                if not self.noblob:
                                    self.write_data([blob_x, blob_y, blob_z], float(self.pars['RBLOB']), superpuntoDatafileName, self.periodic, color=color)

                    if(self.big_sphere):
                        self.write_data(body_pos, self.radius, superpuntoDatafileName, self.periodic, color=16777215)
                    
                    if(self.show_poles):
                        self.write_data(body_pos + np.matmul(R, np.array([0,0,self.radius])), 3*float(self.pars['RBLOB']), superpuntoDatafileName, self.periodic, color=0*65536+0*256+255)
                        self.write_data(body_pos + np.matmul(R, np.array([0,0,-self.radius])), 3*float(self.pars['RBLOB']), superpuntoDatafileName, self.periodic, color=0*65536+0*256+255)

                    for fil in range(int(self.pars['NFIL'])):
                        fil_color = int("000000", base=16)
                        fil_base = body_pos + np.matmul(R, self.fil_references[3*fil : 3*fil+3])
                        if (self.pars['PRESCRIBED_CILIA'] == 1):
                            # WRITE A FUNCTION FOR THIS!!
                            
                            cmap = plt.get_cmap(cmap_name)
                            rgb_color = cmap(fil_phases[fil]/(2*np.pi))[:3]  # Get the RGB color tuple
                            rgb_hex = mcolors.rgb2hex(rgb_color)[1:]  # Convert RGB to BGR hexadecimal format
                            bgr_hex = rgb_hex[4:]+rgb_hex[2:4]+rgb_hex[:2]
                            fil_color = int(bgr_hex, base=16)

                            # ref = self.fillength*R@fitter
                        # self.write_data(fil_base, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)

                        s = np.linspace(0, 1, 20)
                        Rfil = util.rot_mat(self.fil_q[4*fil : 4*fil+4])

                        Rtheta = rotation_matrix = np.array([
                            [np.cos(fil_angles[fil]), -np.sin(fil_angles[fil]), 0],
                            [np.sin(fil_angles[fil]), np.cos(fil_angles[fil]), 0],
                            [0, 0, 1]
                        ])
                        for seg in range(0, int(self.pars['NSEG'])):
                            
                            ref = self.fillength*R@Rfil@Rtheta@np.array(fitted_shape(s[seg], fil_phases[fil]))
                            seg_pos = fil_base + ref
                            self.write_data(seg_pos, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)

                            # if (self.pars['PRESCRIBED_CILIA'] == 1):
                            #     seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)] 
                            # if(self.check_overlap):
                            #     segs_list[fil*self.nseg + seg] = seg_pos
                            # else:
                            #     self.write_data(seg_pos, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)
                            
                if(self.check_overlap):
                    threshold = 1.0
                    cell_size = 10
                    particle_list = np.concatenate([segs_list, blobs_list])

                    colliding_indices, colliding_particles = util.label_colliding_particles_with_3d_cell_list(particle_list, cell_size, threshold*float(self.pars['RSEG']))
                    
                    self.write_data(body_pos, self.radius, superpuntoDatafileName, self.periodic, color=16777215)
                    print(f'Overlapping case at threshold {threshold} = {len(colliding_indices)}')

                    for i, pos in enumerate(segs_list):
                        fil_color = 16777215
                        if(i in colliding_indices):
                            fil_color = 255
                        self.write_data(pos, float(self.pars['RSEG']), superpuntoDatafileName, self.periodic, True, True, color=fil_color)
                        
## Filaments
    def plot_fil(self):
        self.select_sim()

        seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_proj_type('ortho')
        # ax.set_proj_type('persp', 0.05)  # FOV = 157.4 deg
        # ax.view_init(elev=5., azim=45)
        # ax.dist=20
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # ax.axis('off')
        # ax.grid(False)


        def animation_func(t):
            print(t)
            ax.cla()

            ax.set_xlim(-100, 300)
            ax.set_ylim(-100, 300)
            ax.set_zlim(-100, 300)

            seg_states_str = seg_states_f.readline()

            seg_states = np.array(seg_states_str.split()[1:], dtype=float)

            # Robot arm to find segment position (Ignored plane rotation!)
            for fil in range(self.nfil):
                fil_data = np.zeros((self.nseg, 3))
                fil_i = int(3*fil*self.nseg)
                fil_data[0] = seg_states[fil_i : fil_i+3]

                for seg in range(1, self.nseg):
                    seg_pos = seg_states[fil_i+3*(seg-1) : fil_i+3*seg]
                    fil_data[seg] = seg_pos
                ax.plot(fil_data[:,0], fil_data[:,1], fil_data[:,2], c='black', zorder = 100)

        if(self.video):
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            ani = animation.FuncAnimation(fig, animation_func, frames=500, interval=10, repeat=False)
            plt.show()
            # FFwriter = animation.FFMpegWriter(fps=10)
            # ani.save(f'fig/ciliate_{nfil}fil_anim.mp4', writer=FFwriter)
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i==self.plot_end_frame-1):
                    animation_func(i)
                else:
                    seg_states_str = seg_states_f.readline()
                
            plt.savefig(f'fig/ciliate_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
            plt.show()

    def phase_plane(self):

        self.select_sim()
        
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Plotting
        # colormap = 'cividis'
        colormap = 'twilight_shifted'
        colormap = 'hsv'

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fil_references = np.zeros((self.nfil,3))

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        vmin = 0
        vmax = 2*np.pi
        if(self.angle):
            vmin = -.2*np.pi
            vmax = .2*np.pi
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.ax.set_yticks(np.linspace(vmin, vmax, 7), ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        cbar.set_label(r"phase")    

        global frame
        frame = 0
        import scipy.interpolate

        def animation_func(t):
            global frame
            ax.cla()

            fil_states_str = fil_states_f.readline()
            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)

            for i in range(self.nfil):
                fil_references[i] = self.fil_references[3*i: 3*i+3]

            variables = fil_states[:self.nfil]
            if self.angle:
                variables = fil_states[self.nfil:]

            ax.set_title(rf"${frame}$")
            ax.set_ylabel(r"$y$")
            ax.set_xlabel(r"$x$")
            ax.set_aspect('equal')
            # ax.set_xlim(0, 640)
            # ax.set_ylim(0, 2560)
            # ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
            # ax.set_yticks(np.linspace(0, np.pi, 5), ['0', 'π/4', 'π/2', '3π/4', 'π'])
            ax.invert_yaxis()
            fig.tight_layout()

            cmap = mpl.colormaps[colormap]
            colors = cmap(variables/vmax)

            # Interpolation
            if (self.interpolate):
                n1, n2 = 128, 128
                offset = 0.2
                azim_grid = np.linspace(min(fil_references[:,0])+offset, max(fil_references[:,0])-offset, n1)
                polar_grid = np.linspace(min(fil_references[:,1])+offset, max(fil_references[:,1])-offset, n2)
                xx, yy = np.meshgrid(azim_grid, polar_grid)
                xx, yy = xx.ravel(), yy.ravel()

                
                colors_inter = scipy.interpolate.griddata((fil_references[:,0],fil_references[:,1]), colors, (xx, yy), method='linear')
                ax.scatter(xx, yy, c=colors_inter)

                # # Find the region that takes a certain color (say white)
                # white_band = np.where(np.all(colors_inter>np.array([0.75, 0.75, 0.75, 0]), axis=1))
                # mean_x = np.angle(np.mean(np.exp(xx[white_band]*1j)))
                # ax.scatter(mean_x, np.mean(yy[white_band]), c='black', s = 200, marker='s')

                # # Contour (Doesn't work very well because of the discontinuity...)
                # phases_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), variables, (xx, yy), method='linear')
                # xx_grid = np.reshape(xx, (n1, n2))
                # yy_grid = np.reshape(yy, (n1, n2))
                # phases_grid = np.reshape(phases_inter, (n1, n2))
                # levels = np.linspace(0, 2*np.pi, 3)[1:-1]
                # contour = ax.contour(xx_grid, yy_grid, phases_grid, levels=levels, cmap='hsv')
                # ax.clabel(contour, levels)
                # # Find centeroid of contours
                # for i, line in enumerate(contour.collections):
                #     paths = line.get_paths()
                #     centers_of_contour = list()
                #     lengths = [len(path) for path in paths]
                #     max_length = max(lengths)
                #     for p, path in enumerate(paths):
                #         if(len(path)==max_length):
                #             points = path.vertices
                #             center = np.mean(points, axis=0)
                #             centers_of_contour.append(center)
                #             ax.scatter(center[0], center[1], s = 100)
                #             break
                        
            else:
            # Individual filaments
                ax.scatter(fil_references[:,0], fil_references[:,1], c=colors)
                
                # # Find the region that takes a certain color (say white)
                # white_band = np.where(np.all(colors>np.array([0.75, 0.75, 0.75, 0]), axis=1))
                # ax.scatter(fil_references_sphpolar[:,1][white_band], fil_references_sphpolar[:,2][white_band], marker='x')
                # mean_x = np.angle(np.mean(np.exp(fil_references_sphpolar[:,1][white_band]*1j)))
                # ax.scatter(mean_x, np.mean(fil_references_sphpolar[:,2][white_band]), c='black', s = 200, marker='s')
                

            frame += 1

        if(self.video):
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i>=self.plot_start_frame):
                    frame = i
                    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                    ani = animation.FuncAnimation(fig, animation_func, frames=self.frames, interval=10, repeat=False)
                    plt.show()    
                    # FFwriter = animation.FFMpegWriter(fps=16)
                    # ani.save(f'fig/fil_phase_index{self.index}_{self.date}_anim.mp4', writer=FFwriter)
                    ## when save, need to comment out plt.show() and be patient!
                    break
                else:
                    # fil_phases_str = fil_phases_f.readline()
                    fil_states_str = fil_states_f.readline()
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i==self.plot_end_frame-1):
                    animation_func(i)
                else:
                    fil_states_str = fil_states_f.readline()
                    # fil_phases_str = fil_phases_f.readline()
                    # if(self.angle):
                    #     fil_angles_str = fil_angles_f.readline()
                    frame += 1
                
            plt.savefig(f'fig/fil_phase_plane_index{self.index}_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
            plt.show()

## Ciliates
# Single sim
    def phase(self):

        self.select_sim()
        
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Plotting
        # colormap = 'cividis'
        colormap = 'twilight_shifted'
        # colormap = 'hsv'

        # colormap = 'binary'

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fil_references_sphpolar = np.zeros((self.nfil,3))
        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])


        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        vmin = 0
        vmax = 2*np.pi
        if(self.angle):
            vmin = -.2*np.pi
            vmax = .2*np.pi
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.ax.set_yticks(np.linspace(vmin, vmax, 7), ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        cbar.set_label(r"phase")    

        global frame
        frame = 0
        import scipy.interpolate

        def animation_func(t):
            global frame
            ax.cla()

            fil_states_str = fil_states_f.readline()
            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)

            
            variables = fil_states[:self.nfil]
            if self.angle:
                variables = fil_states[self.nfil:]

            ax.set_title(rf"${frame}$")
            ax.set_ylabel(r"$\theta$")
            ax.set_xlabel(r"$\phi$")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0, np.pi)
            ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
            ax.set_yticks(np.linspace(0, np.pi, 5), ['0', 'π/4', 'π/2', '3π/4', 'π'])
            ax.invert_yaxis()
            fig.tight_layout()

            cmap = mpl.colormaps[colormap]
            colors = cmap(variables/vmax)

            # Interpolation
            if (self.interpolate):
                n1, n2 = 128, 128
                offset = 0.2
                azim_grid = np.linspace(min(fil_references_sphpolar[:,1])+offset, max(fil_references_sphpolar[:,1])-offset, n1)
                polar_grid = np.linspace(min(fil_references_sphpolar[:,2])+offset, max(fil_references_sphpolar[:,2])-offset, n2)
                xx, yy = np.meshgrid(azim_grid, polar_grid)
                xx, yy = xx.ravel(), yy.ravel()

                
                colors_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), colors, (xx, yy), method='linear')
                ax.scatter(xx, yy, c=colors_inter)

                # # Find the region that takes a certain color (say white)
                # white_band = np.where(np.all(colors_inter>np.array([0.75, 0.75, 0.75, 0]), axis=1))
                # mean_x = np.angle(np.mean(np.exp(xx[white_band]*1j)))
                # ax.scatter(mean_x, np.mean(yy[white_band]), c='black', s = 200, marker='s')

                # # Contour (Doesn't work very well because of the discontinuity...)
                # phases_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), variables, (xx, yy), method='linear')
                # xx_grid = np.reshape(xx, (n1, n2))
                # yy_grid = np.reshape(yy, (n1, n2))
                # phases_grid = np.reshape(phases_inter, (n1, n2))
                # levels = np.linspace(0, 2*np.pi, 3)[1:-1]
                # contour = ax.contour(xx_grid, yy_grid, phases_grid, levels=levels, cmap='hsv')
                # ax.clabel(contour, levels)
                # # Find centeroid of contours
                # for i, line in enumerate(contour.collections):
                #     paths = line.get_paths()
                #     centers_of_contour = list()
                #     lengths = [len(path) for path in paths]
                #     max_length = max(lengths)
                #     for p, path in enumerate(paths):
                #         if(len(path)==max_length):
                #             points = path.vertices
                #             center = np.mean(points, axis=0)
                #             centers_of_contour.append(center)
                #             ax.scatter(center[0], center[1], s = 100)
                #             break
                        
            else:
            # Individual filaments
                ax.scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)
                
                # # Find the region that takes a certain color (say white)
                # white_band = np.where(np.all(colors>np.array([0.75, 0.75, 0.75, 0]), axis=1))
                # ax.scatter(fil_references_sphpolar[:,1][white_band], fil_references_sphpolar[:,2][white_band], marker='x')
                # mean_x = np.angle(np.mean(np.exp(fil_references_sphpolar[:,1][white_band]*1j)))
                # ax.scatter(mean_x, np.mean(fil_references_sphpolar[:,2][white_band]), c='black', s = 200, marker='s')
                

            frame += 1

        if(self.video):
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i>=self.plot_start_frame):
                    frame = i
                    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                    ani = animation.FuncAnimation(fig, animation_func, frames=self.frames, interval=10, repeat=False)
                    plt.show()    
                    # FFwriter = animation.FFMpegWriter(fps=16)
                    # ani.save(f'fig/fil_phase_index{self.index}_{self.date}_anim.mp4', writer=FFwriter)
                    ## when save, need to comment out plt.show() and be patient!
                    break
                else:
                    # fil_phases_str = fil_phases_f.readline()
                    fil_states_str = fil_states_f.readline()
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i==self.plot_end_frame-1):
                    animation_func(i)
                else:
                    fil_states_str = fil_states_f.readline()
                    # fil_phases_str = fil_phases_f.readline()
                    # if(self.angle):
                    #     fil_angles_str = fil_angles_f.readline()
                    frame += 1
                
            plt.savefig(f'fig/fil_phase_index{self.index}_{self.date}_frame{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
            plt.show()

    def wavenumber(self):

        self.select_sim()
        
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        colormap = 'Greys'

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)

        fil_references_sphpolar = np.zeros((self.nfil,3))
        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])


        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        vmin = 0
        vmax = 2*np.pi
        
        import scipy.interpolate

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        avg_posterior_phase_array = np.zeros(self.frames)
        avg_anterior_phase_array = np.zeros(self.frames)
        fil_phases_ref = np.zeros(self.nfil)

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_states_str = fil_states_f.readline()

            if(i>=self.plot_start_frame):
            # if(i==self.plot_end_frame-1):
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                fil_phases = fil_states[:self.nfil]
                fil_phases_boxed = util.box(fil_phases, 2*np.pi)

                if(i==self.plot_start_frame):
                    fil_phases_ref = fil_phases - fil_phases_boxed
                    
                
                fil_phases -= fil_phases_ref
                

                cmap = mpl.colormaps[colormap]
                colors = cmap(fil_phases_boxed/vmax)

                n1, n2 = 128, 128
                offset = 0.2
                azim_grid = np.linspace(min(fil_references_sphpolar[:,1])+offset, max(fil_references_sphpolar[:,1])-offset, n1)
                polar_grid = np.linspace(min(fil_references_sphpolar[:,2])+offset, max(fil_references_sphpolar[:,2])-offset, n2)
                xx, yy = np.meshgrid(azim_grid, polar_grid)
                xx, yy = xx.ravel(), yy.ravel()

                colors_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), colors, (xx, yy), method='linear')
                # ax.scatter(xx, yy, c=colors_inter)

                phases_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), fil_phases, (xx, yy), method='nearest')
                phases_inter_boxed = util.box(phases_inter, 2*np.pi)
                colors_new = cmap(phases_inter_boxed/vmax)
                # if(i==self.plot_end_frame-1):
                if(i==self.plot_start_frame):
                    ax.scatter(xx, yy, c=colors_new)

                avg_posterior_phase = np.mean(phases_inter[:128])
                avg_anterior_phase = np.mean(phases_inter[-128:])
                print(i, avg_posterior_phase, avg_anterior_phase)
                avg_posterior_phase_array[i-self.plot_start_frame] = avg_posterior_phase
                avg_anterior_phase_array[i-self.plot_start_frame] = avg_anterior_phase

        wavenumber_array = (avg_posterior_phase_array - avg_anterior_phase_array)/(2*np.pi) +1 
        ax2.plot(time_array, wavenumber_array, c='black')
        # ax2.plot(time_array, avg_anterior_phase_array, c='r')
        # ax2.plot(time_array, avg_posterior_phase_array, c='b')

        np.save(f'{self.dir}/time_array_fil{self.nfil}.npy', time_array)
        np.save(f'{self.dir}/wavenumber_array_fil{self.nfil}.npy', wavenumber_array)

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, np.pi)
        
        fig.tight_layout()
        fig2.tight_layout()
        plt.show()

    def phi_dot(self):
        self.select_sim()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(1,1,1)
        fil_references_sphpolar = np.zeros((self.nfil,3))
        
        fil_phases_f = open(self.simName + '_filament_phases.dat', "r")
        fil_angles_f = open(self.simName + '_filament_shape_rotation_angles.dat', "r")


        nfil = self.nfil
        n_snapshots = min(300, self.plot_end_frame)
        start = self.plot_end_frame - n_snapshots
        X = np.zeros((nfil, n_snapshots))
        time_array = np.arange(n_snapshots)

        fil_references_sphpolar = np.zeros((nfil,3))
        for fil in range(nfil):
            fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
        azim_array = fil_references_sphpolar[:,1]
        polar_array = fil_references_sphpolar[:,2]
        sorted_indices = np.argsort(azim_array)
        azim_array_sorted = azim_array[sorted_indices]
        polar_array_sorted = polar_array[sorted_indices]
        fil_phases_sorted = np.array([])
        

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_phases_str = fil_phases_f.readline()
            fil_angles_str = fil_angles_f.readline()
            
            if(i>=start):
                fil_phases = np.array(fil_phases_str.split()[1:], dtype=float)
                fil_phases_sorted = fil_phases[sorted_indices]

                fil_angles = np.array(fil_angles_str.split()[1:], dtype=float)
                fil_angles_sorted = fil_angles[sorted_indices]

                X[:,i-start] = fil_phases_sorted[:nfil]
                # X[:,i-start] = fil_angles_sorted[:nfil]
        

        phi_unboxed = X
        phi = util.box(phi_unboxed, 2*np.pi)
        phi_dot = np.diff(phi_unboxed)
        ax4.scatter(azim_array_sorted, polar_array_sorted, c = phi_dot[:,-1])

        r = np.abs(np.sum(np.exp(phi*1j), 0)/self.nfil)

        diff = np.diff(np.sin(phi), axis=0)
        corr_matrix = np.abs(diff[:-1,:]) + np.abs(diff[1:,:])
        corr_over_time = np.mean(corr_matrix, 0)
        avg_corr = np.mean(corr_over_time)
        print(f"index={self.index} correlation={avg_corr}")
        

        for i in range(5):
            ax.plot(time_array[1:], phi_dot[i,:])
            ax2.plot(time_array, phi[i,:])
            ax3.plot(time_array, corr_over_time)
        ax.set_xlabel('time')
        
        #########################
        # Combine x and y into a single array
        data = np.column_stack((polar_array_sorted, azim_array_sorted))

        # Specify the number of clusters you want
        n_clusters = int(self.nfil/10) 
        corr_array = np.zeros(n_clusters) # correlation within each cluster

        # Create and fit a K-Means model
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        cluster_assignments = kmeans.labels_

        for i in range(n_clusters):
            phases_in_group = np.sin(fil_phases_sorted[np.where(cluster_assignments==i)])
            corr_array[i] = np.var(phases_in_group)
        
        avg_var = np.mean(corr_array)
        print(f"index={self.index} variance={avg_var}")
    
        ##################
            
        ax5.scatter(azim_array_sorted, polar_array_sorted, c = cluster_assignments)
        

        fig.savefig(f'fig/fil_order_parameter_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()
      
    def order_parameter(self):
        self.select_sim()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(1,1,1)
        # fig4 = plt.figure()
        # ax4 = fig4.add_subplot(1,1,1)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(1,1,1)
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(1,1,1)

        fil_states_f = open(self.simName + '_true_states.dat', "r")

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        corr_array = np.zeros(self.frames)
        corr_array2 = np.zeros(self.frames)
        corr_array_angle = np.zeros(self.frames)
        r_array = np.zeros(self.frames)
        wavenumber_array = np.zeros(self.frames)
        effective_beat_array = np.zeros(self.frames)

        fil_references_sphpolar = np.zeros((self.nfil,3))
        for fil in range(self.nfil):
            fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
        azim_array = fil_references_sphpolar[:,1]
        polar_array = fil_references_sphpolar[:,2]
        sorted_indices = np.argsort(polar_array)
        azim_array_sorted = azim_array[sorted_indices]
        polar_array_sorted = polar_array[sorted_indices]
        fil_phases_sorted = np.array([])
        fil_angles_sorted = np.array([])

        #########################
        # Combine x and y into a single array
        pos_data = np.column_stack((polar_array_sorted, azim_array_sorted))

        # Specify the number of clusters you want
        n_clusters = int(self.nfil/10) 
        variance_array = np.zeros(n_clusters) # correlation within each cluster
        variance_array_angle = np.zeros(n_clusters) # correlation within each cluster

        # Create and fit a K-Means model
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pos_data)
        cluster_assignments = kmeans.labels_
        ##################

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_states_str = fil_states_f.readline()

            if(i>=self.plot_start_frame):
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                fil_phases = fil_states[:self.nfil]
                fil_phases = util.box(fil_phases, 2*np.pi)
                fil_angles = fil_states[self.nfil:]

                fil_angles_sorted = fil_angles[sorted_indices]
                fil_phases_sorted = fil_phases[sorted_indices]
                sin_phases_sorted = np.sin(fil_phases_sorted)

                # Coordination number 1
                phase_diff = np.diff(sin_phases_sorted, prepend=sin_phases_sorted[-1])
                corr = np.abs(phase_diff[:-1]) + np.abs(phase_diff[1:])
                corr_array[i-self.plot_start_frame] = np.mean(corr)

                # Coordination number 2
                for m in range(n_clusters):
                    phases_in_group = sin_phases_sorted[np.where(cluster_assignments==m)]
                    variance_array[m] = np.var(phases_in_group)
                
                corr_array2[i-self.plot_start_frame] = np.mean(variance_array)

                # Coordination number for angle
                for m in range(n_clusters):
                    angles_in_group = fil_angles_sorted[np.where(cluster_assignments==m)]
                    variance_array_angle[m] = np.var(angles_in_group)
                
                corr_array_angle[i-self.plot_start_frame] = np.mean(variance_array_angle)
                r_array[i-self.plot_start_frame] = np.abs(np.sum(np.exp(1j*fil_phases))/self.nfil)
                wavenumber_array[i-self.plot_start_frame] = fil_phases_sorted[0] - fil_phases_sorted[-1]
                effective_beat_array[i-self.plot_start_frame] = len([phase for phase in fil_phases if 0.1 <= phase <= 1.3])

        ax.plot(time_array, corr_array)
        ax.set_xlabel('t/T')
        ax.set_ylabel('Coordination number')
        ax.set_xlim(time_array[0], time_array[-1])
        ax.set_ylim(0)

        ax2.plot(time_array, corr_array2)
        ax2.set_xlabel('t/T')
        ax2.set_ylabel('Coordination number 2')
        ax2.set_xlim(time_array[0], time_array[-1])
        ax2.set_ylim(0)
        
        # ax3.scatter(azim_array_sorted, polar_array_sorted, c = cluster_assignments)
        # ax3.set_xlabel(r'$\phi$')
        # ax3.set_ylabel(r'$\theta$')

        # ax4.plot(time_array, corr_array_angle)
        # ax4.set_xlabel('t/T')
        # ax4.set_ylabel('Coordination number 2 (angle)')
        # ax4.set_xlim(time_array[0], time_array[-1])
        # ax4.set_ylim(0)

        ax5.plot(time_array, r_array)
        ax5.set_ylim(0)
        ax5.set_xlabel('t/T')
        ax5.set_ylabel('<r>')
        ax5.set_xlim(time_array[0], time_array[-1])
        # ax5.set_xticks(np.linspace(0, 40, 5))

        ax6.plot(time_array, effective_beat_array)
        ax6.set_xlabel('t/T')
        ax6.set_ylabel('No. of effective strokes')

        np.save(f'{self.dir}/time_array_index{self.index}.npy', time_array)
        np.save(f'{self.dir}/r_array_index{self.index}.npy', r_array)
        np.save(f'{self.dir}/num_eff_beat_array_index{self.index}.npy', effective_beat_array)
        
        # fig.savefig(f'fig/fil_coordination_parameter_one_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/fil_coordination_parameter_two_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig3.savefig(f'fig/fil_clustering_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        fig.tight_layout()
        fig2.tight_layout()
        fig5.tight_layout()
        fig6.tight_layout()
        fig5.savefig(f'fig/oder_parameter_index{self.index}.png', bbox_inches = 'tight', format='png')
        fig6.savefig(f'fig/num_effective_stroke_index{self.index}.png', bbox_inches = 'tight', format='png')
        plt.show()

    def footpath(self):
        self.select_sim()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)

        fil_states_f = open(self.simName + '_true_states.dat', "r")

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        r_array = np.zeros(self.frames)
        corr_array = np.zeros(self.frames)
        corr_array2 = np.zeros(self.frames)
        states_array = np.zeros((self.frames, self.nfil))
        states_array2 = np.zeros((self.frames, self.nfil))

        window_size = 1
        windowd_length = self.frames - window_size + 1
        r_avg_array = np.zeros(windowd_length)

        fil_references_sphpolar = np.zeros((self.nfil,3))
        fil_references_cartersian = np.zeros((self.nfil,3))
        for fil in range(self.nfil):
            fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
            fil_references_cartersian[fil] = self.fil_references[3*fil: 3*fil+3]
        azim_array = fil_references_sphpolar[:,1]
        polar_array = fil_references_sphpolar[:,2]
        sorted_indices = np.argsort(polar_array)
        azim_array_sorted = azim_array[sorted_indices]
        polar_array_sorted = polar_array[sorted_indices]

        #########################
        # Combine x and y into a single array
        pos_data = np.column_stack((polar_array_sorted, azim_array_sorted))

        # Specify the number of clusters you want
        n_clusters = int(self.nfil/10) 
        variance_array = np.zeros(n_clusters) # correlation within each cluster
        variance_array_angle = np.zeros(n_clusters) # correlation within each cluster

        # Create and fit a K-Means model
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pos_data)
        cluster_assignments = kmeans.labels_
        ##################

        # Create a mapping to its nearest filament
        from scipy.spatial.distance import cdist
        def nearest_particles(positions):
            distances = cdist(positions, positions)
            np.fill_diagonal(distances, np.inf)
            nearest_indices = np.argmin(distances, axis=1)
            return nearest_indices
        nearest_indices = nearest_particles(fil_references_cartersian)

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_states_str = fil_states_f.readline()

            if(i>=self.plot_start_frame):
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                fil_phases = fil_states[:self.nfil]
                fil_phases = util.box(fil_phases, 2*np.pi)
                # fil_angles = fil_states[self.nfil:]

                # fil_angles_sorted = fil_angles[sorted_indices]
                # fil_phases_sorted = fil_phases[sorted_indices]
                # sin_phases_sorted = np.sin(fil_phases_sorted)

                # # Coordination number 1
                # phase_diff = np.diff(sin_phases_sorted, prepend=sin_phases_sorted[-1])
                # corr = np.abs(phase_diff[:-1]) + np.abs(phase_diff[1:])
                # corr_array[i-self.plot_start_frame] = np.mean(corr)

                diff = np.sin(fil_phases) - np.sin(fil_phases[nearest_indices])
                # print(diff)
                corr_array[i-self.plot_start_frame] = np.mean(np.abs(diff))

                r_array[i-self.plot_start_frame] = np.abs(np.sum(np.exp(1j*fil_phases))/self.nfil)

                # states_array[i-self.plot_start_frame] = fil_phases_sorted
                # states_array2[i-self.plot_start_frame] = fil_phases_sorted[nearest_indices[sorted_indices]]

                # states_array[i-self.plot_start_frame] = fil_phases
                # states_array2[i-self.plot_start_frame] = fil_phases[nearest_indices]


        
        for i in range(windowd_length):
            r_avg_array[i] = np.mean(r_array[i:i+window_size])
        ax.plot(time_array[:windowd_length], r_avg_array)
        ax.set_xlabel('t/T')
        ax.set_ylabel(r'$<r>$')
        ax.set_xlim(time_array[0], time_array[-1])
        ax.set_ylim(0)


        # ax2.plot(states_array[:, 120], states_array[:,1])

        # print(states_array[0, :20])
        # print(states_array2[0, :20])


        ax2.plot(time_array, corr_array)
        ax2.set_xlabel('t/T')
        ax2.set_ylabel(r'Synchronisation number')
        # ax2.set_xlim(time_array[0], time_array[-1])
        # ax2.set_ylim(0)

        ax3.plot(corr_array[:windowd_length], r_avg_array)
        ax3.set_xlim(0)
        ax3.set_ylim(0)
        ax3.set_xlabel(r'Synchronisation number')
        ax3.set_ylabel(r'$<r>$')
        

        
        fig.savefig(f'fig/fil_order_parameter_index{self.index}_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/fil_coordination_parameter_two_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig3.savefig(f'fig/fil_clustering_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def kymograph(self):
        self.select_sim()

        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Plotting
        colormap = 'cividis'
        colormap = 'twilight_shifted'
        # colormap = 'jet'

        # fig = plt.figure()
        fig, axs = plt.subplots(2, 1, sharex=True)
        fil_references_sphpolar = np.zeros((self.nfil,3))

        import scipy.interpolate

        n1, n2 = 200, 100 
        azim_array = np.linspace(-np.pi, np.pi, n1)
        polar_array = np.linspace(0, np.pi, n2)

        phi_kymo = np.zeros((n1, self.frames))
        theta_kymo = np.zeros((n2, self.frames))

        time_array = (np.arange(self.frames) + self.plot_start_frame)/self.period
        
        phi_kymo_xx, phi_kymo_yy = np.meshgrid(time_array, azim_array,)
        theta_kymo_xx, theta_kymo_yy = np.meshgrid(time_array, polar_array, )
        
        for m in range(self.nfil):
            fil_references_sphpolar[m] = util.cartesian_to_spherical(self.fil_references[3*m: 3*m+3])
        
        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_states_str = fil_states_f.readline()
            if(i>=self.plot_start_frame):
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)
                fil_phases = fil_states[:self.nfil]
                fil_angles = fil_states[self.nfil:]

                if self.angle:
                    variables = fil_angles
                else:
                    variables = fil_phases

                # Interpolation
                
                xx, yy = np.meshgrid(azim_array, polar_array)
                zz = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), variables, (xx, yy), method='nearest')

                phi_kymo[:, i-self.plot_start_frame] = zz[n2//2]
                theta_kymo[:, i-self.plot_start_frame] = zz[:, n1//2]
            

        axs[0].scatter(phi_kymo_xx, phi_kymo_yy, c=phi_kymo, cmap=colormap, vmin=0, vmax=2*np.pi)
        axs[1].scatter(theta_kymo_xx, theta_kymo_yy, c=theta_kymo, cmap=colormap, vmin=0, vmax=2*np.pi)       
            
        # axs[0].set_xlabel(r"$t$")
        axs[0].set_ylabel(r"$\phi$")
        axs[0].set_xlim(time_array[0], time_array[-1])
        axs[0].set_ylim(-np.pi, np.pi)
        axs[0].set_yticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0π', 'π/2', 'π'])
        
        axs[1].set_xlabel(r"$t/T$")
        axs[1].set_ylabel(r"$\theta$")
        axs[1].set_xlim(time_array[0], time_array[-1])
        axs[1].set_ylim(0, np.pi)
        axs[1].invert_yaxis()
        axs[1].set_yticks(np.linspace(0, np.pi, 5), ['0', 'π/4', 'π/2', '3π/4', 'π'])

        fig.savefig(f'fig/kymograph_index{self.index}_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def eckert(self):
        R = 1
        phi0 = np.pi
        from scipy import optimize
        def find_k(k, theta):
            return k + np.sin(k)*np.cos(k) + 2*np.sin(k) - (2 + np.pi/2)*np.sin(theta)

        def eckert_projection(theta, phi):
            sign = 1
            if theta >= np.pi/2:
                theta = (np.pi/2 - (theta - np.pi/2))
                sign = -1
            
            k=optimize.newton(find_k, 1, args=(theta,))

            x = 0.4222382*R*(phi-phi0)*(1+np.cos(k))
            y = 1.3265004*R*np.sin(k)*sign
            return x, y
    
        self.select_sim()
        
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Plotting
        colormap = 'cividis'
        colormap = 'twilight_shifted'

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fil_references_sphpolar = np.zeros((self.nfil,3))

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=0, vmax=2*np.pi)
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.ax.set_yticks(np.linspace(0, 2*np.pi, 7), ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        cbar.set_label(r"phase")

        ax.set_ylabel(r"$\theta$")
        ax.set_xlabel(r"$\phi$")
        # ax.set_xlim(-np.pi, np.pi)
        # ax.set_ylim(0, np.pi)
        ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
        ax.set_yticks(np.linspace(0, np.pi, 5), ['0', 'π/4', 'π/2', '3π/4', 'π'])


        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])

        global frame
        frame = 0

        def animation_func(t):
            global frame

            ax.cla()
            fil_states_str = fil_states_f.readline()

            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)

            fil_phases = fil_states[:self.nfil]
                
            projected_points = [eckert_projection(theta, phi) for theta, phi in zip(fil_references_sphpolar[:,2], fil_references_sphpolar[:,1])]
            projected_x, projected_y = zip(*projected_points)
            
            ax.scatter(projected_x, projected_y, c=fil_phases, cmap=colormap, vmin=0, vmax=2*np.pi)
            frame += 1

        if(self.video):
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i>=self.plot_start_frame):
                    frame = i
                    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                    ani = animation.FuncAnimation(fig, animation_func, frames=500, interval=10, repeat=False)
                    plt.show()
                    FFwriter = animation.FFMpegWriter(fps=10)
                    ani.save(f'fig/fil_phase_{self.nfil}fil_anim.mp4', writer=FFwriter)
                    break
                else:
                    fil_states_str = fil_states_f.readline()
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i==self.plot_end_frame-1):
                    animation_func(i)
                else:
                    fil_states_str = fil_states_f.readline()

            plt.savefig(f'fig/fil_phase_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
            plt.show()
     
    def spherical_contour(self):
        self.select_sim()
        
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Plotting
        colormap = 'twilight_shifted'
        colormap = 'hsv'

        fig = plt.figure()
        ax = fig.add_subplot(111)

        fil_references_sphpolar = np.zeros((self.nfil,3))

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        vmin = 0
        vmax = 2*np.pi
        if(self.angle):
            vmin = -.2*np.pi
            vmax = .2*np.pi
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # sm = ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm)
        # cbar.ax.set_yticks(np.linspace(vmin, vmax, 7), ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        # cbar.set_label(r"phase")    
        
        global frame
        frame = 0
        import scipy.interpolate

        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_states_str = fil_states_f.readline()
            if(i==self.plot_end_frame-1):
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)

                variables = fil_states[:self.nfil]
                if self.angle:
                    variables = fil_states[self.nfil:]
                
                # u: azim v: polar
                u = fil_references_sphpolar[:,1]
                v = fil_references_sphpolar[:,2]
                # Plot only a hemisphere
                min_phi, max_phi = 0, np.pi
                hemisphere_indices = np.where((min_phi<u) & (max_phi>u))


                u = u[hemisphere_indices]
                v = v[hemisphere_indices]
                variables = variables[hemisphere_indices]

                R  = fil_references_sphpolar[0,0]

                x,y,z = util.spherical_to_cartesian(R, u, v)

                def generate(r, n):
                    Ntotal = int(n*6 + n*(n-1)*6/2)
                    dR = r/n
        
                    r = np.zeros(Ntotal)
                    p = np.zeros(Ntotal)
                    for l in range(n):
                        start = int(6*l + l*(l-1)*6/2)
                        Np = int((l+1)*6)
                        dp = 2*np.pi/Np
                        for point in range(Np):
                            r[start + point] = dR*(l+1)
                            p[start + point] = point*dp

                    r = np.insert(r, 0, 0)
                    p = np.insert(p, 0, 0)
                    return r, p

                nring = 30
                offset = 0.9
                r, p = generate(R*offset, nring)
                xx, zz = r*np.sin(p), r*np.cos(p)

                
                # ax.scatter(x, z, c=variables, cmap=colormap, vmin=vmin, vmax=vmax)

                cmap = mpl.colormaps[colormap]
                colors = cmap(variables/vmax)
                colors = scipy.interpolate.griddata((x, z), colors, (xx, zz), method='nearest')

                ax.scatter(xx, zz, c=colors)

        ax.set_ylabel(r"$\theta$")
        ax.set_xlabel(r"$\phi$")
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect('equal')

        ax.axis('off')

        fig.tight_layout()
        fig.savefig(f'fig/fil_phase_sph_index{self.index}_{self.date}_{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def ciliate(self):
        show_flow_field = False
        self.select_sim()

        def stokeslet(x, x0, f0):
            r = np.linalg.norm(x-x0)
            if r == 0:
                return np.zeros(3)
            return f0/(8.*np.pi*r) + np.dot(f0, x-x0) * (x - x0) / (8.*np.pi*r**3)

        @cuda.jit(device=True)
        def stokeslet_device(x, x0, f0, result):
            dis_x = x[0] - x0[0]
            dis_y = x[1] - x0[1]
            dis_z = x[2] - x0[2]
            r = (dis_x**2 + dis_y**2 + dis_z**2)**0.5
            
            coeff = 1 / (8. * np.pi * r)
            dot_product = f0[0] * dis_x + f0[1] * dis_y + f0[2] * dis_z
            vector_term_x = dot_product * dis_x / (8. * np.pi * r**3)
            vector_term_y = dot_product * dis_y / (8. * np.pi * r**3)
            vector_term_z = dot_product * dis_z / (8. * np.pi * r**3)
            
            result[0] = coeff*f0[0] + vector_term_x
            result[1] = coeff*f0[1] + vector_term_y
            result[2] = coeff*f0[2] + vector_term_z

        @cuda.jit
        def compute_v_list_kernel(pos_list, source_pos_list, source_force_list, v_list):
            pi = cuda.grid(1)
            if pi < pos_list.shape[0]:
                pos = pos_list[pi]
                v_temp = cuda.local.array(3, float64)  # Local array for temporary storage
                v_temp[0] = 0.0
                v_temp[1] = 0.0
                v_temp[2] = 0.0
                for si in range(source_pos_list.shape[0]):
                    source_pos = source_pos_list[si]
                    source_force = source_force_list[si]
                    temp_result = cuda.local.array(3, dtype=float64)
                    stokeslet_device(pos, source_pos, source_force, temp_result)
                    v_temp[0] += temp_result[0]
                    v_temp[1] += temp_result[1]
                    v_temp[2] += temp_result[2]
                v_list[pi, 0] += v_temp[0]
                v_list[pi, 1] += v_temp[1]
                v_list[pi, 2] += v_temp[2]

        if show_flow_field:
            seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
            blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
        seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Create the sphere data points
        num_points = 300
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)
        x = self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_proj_type('ortho')
        ax.view_init(elev=0., azim=90)
        ax.dist=7.4        

        # Flow field
        n_theta = 20
        n_r = 1
        n_phi = 2
        n_field_point = n_theta*n_r*n_phi

        r_list = np.linspace(1.3, 2.0, n_r)*self.radius
        theta_list = np.linspace(0, np.pi, n_theta)
        phi_list = np.linspace(0, 2*np.pi, n_phi+1)[:-1]

        R, Phi, Theta = np.meshgrid(r_list, phi_list, theta_list, indexing='ij')

        r_flat = R.ravel()
        theta_flat = Theta.ravel()
        phi_flat = Phi.ravel()

        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)

        x_flat = X.ravel()
        y_flat = Y.ravel()
        z_flat = Z.ravel()

        pos_list = np.column_stack((x_flat, y_flat, z_flat))
        

        cmap_name = 'hsv'
        cmap = plt.get_cmap(cmap_name)

        def animation_func(t):
            print(t)
            ax.cla()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            v_list = np.zeros(np.shape(pos_list))

            if show_flow_field:
                seg_forces_str = seg_forces_f.readline()
                blob_forces_str = blob_forces_f.readline()
            seg_states_str = seg_states_f.readline()
            body_states_str = body_states_f.readline()
            fil_states_str = fil_states_f.readline()

            if show_flow_field:
                seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
            seg_states = np.array(seg_states_str.split()[1:], dtype=float)
            body_states = np.array(body_states_str.split()[1:], dtype=float)
            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)
            fil_phases = fil_states[:self.nfil]

            
            for swim in range(self.nswim):
                # blob_data = np.zeros((int(self.pars['NBLOB']), 3))
                body_pos = body_states[7*swim : 7*swim+3]
                R = util.rot_mat(body_states[7*swim+3 : 7*swim+7])
                # Plot the sphere
                ax.plot_surface(x+body_pos[0], y+body_pos[1], z+body_pos[2], color='grey', alpha=0.5)

                # Robot arm to find segment position (Ignored plane rotation!)
                for fil in range(self.nfil):
                    fil_data = np.zeros((self.nseg, 3))
                    fil_i = int(3*fil*self.nseg)
                    print(" fil ", fil, "          ", end="\r")

                    fil_color = cmap(fil_phases[fil]/(2*np.pi))

                    for seg in range(self.nseg):
                        seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)]
                        fil_data[seg] = seg_pos

                        if(show_flow_field):
                            seg_force = seg_forces[2*fil_i+6*(seg) : 2*fil_i+6*(seg+1)]
                            seg_force = seg_force[:3]
                            for pi, pos in enumerate(pos_list):
                                v_list[pi] += stokeslet(pos, seg_pos, seg_force)
                    
                    ax.plot(fil_data[:,0], fil_data[:,1], fil_data[:,2], c=fil_color, zorder = 100)

                if(show_flow_field):
                    for blob in range(int(self.pars['NBLOB'])):
                        print(" blob ", blob, "          ", end="\r")
                        blob_pos = np.array(util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3]))
                        blob_force = blob_forces[3*blob : 3*blob+3]
                        for pi, pos in enumerate(pos_list):
                            v_list[pi] += stokeslet(pos, blob_pos, blob_force)
                    
                    colormap2 = 'jet'
                    cmap2 = mpl.colormaps[colormap2]
                    speed_list = np.linalg.norm(v_list, axis=1)

                    # ur_list = v_list[:, 0] * np.sin(theta_flat) * np.cos(phi_flat) + \
                    #             v_list[:, 1] * np.sin(theta_flat) * np.sin(phi_flat) + \
                    #                 v_list[:, 2] * np.cos(theta_flat)
                    # uphi_list = - v_list[:, 0] * np.sin(phi_flat) + v_list[:, 1] * np.cos(phi_flat)
                    # utheta_list = v_list[:, 0] * np.cos(theta_flat) * np.cos(phi_flat) + \
                    #                 v_list[:, 1] * np.cos(theta_flat) * np.sin(phi_flat) \
                    #                     - v_list[:, 2] * np.sin(theta_flat)

                    max_speed = max(speed_list)
                    print('maxspeed', max_speed)
                    colors = cmap2(np.clip(speed_list/35, None, 0.999))
                    ax.scatter(pos_list[:,0], pos_list[:,1], pos_list[:,2], color=colors)
                    ax.quiver(pos_list[:,0], pos_list[:,1], pos_list[:,2], v_list[:,0], v_list[:,1], v_list[:,2], length = 4.5, color='b' )

                    # # e_r
                    # ax.quiver(pos_list[:,0], pos_list[:,1], pos_list[:,2], \
                    #           np.sin(theta_flat)*np.cos(phi_flat)*ur_list,
                    #           np.sin(theta_flat)*np.sin(phi_flat)*ur_list,
                    #           np.cos(theta_flat)*ur_list, length = .5 ,color='r')
                    # # e_phi
                    # ax.quiver(pos_list[:,0], pos_list[:,1], pos_list[:,2], \
                    #           -np.sin(phi_flat)*uphi_list,
                    #           np.cos(phi_flat)*uphi_list,
                    #           np.zeros(n_field_point), length = .5 ,color='r')
                    # # e_theta
                    # ax.quiver(pos_list[:,0], pos_list[:,1], pos_list[:,2], \
                    #           np.cos(theta_flat)*np.cos(phi_flat)*utheta_list,
                    #           np.cos(theta_flat)*np.sin(phi_flat)*utheta_list,
                    #           -np.sin(theta_flat)*utheta_list, length = .5 ,color='r')
                    

        if(self.video):
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i>=self.plot_start_frame):
                    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                    ani = animation.FuncAnimation(fig, animation_func, frames=500, interval=10, repeat=False)
                    # plt.show()
                    break
                else:
                    if show_flow_field:
                        seg_forces_str = seg_forces_f.readline()
                        blob_forces_str = blob_forces_f.readline()
                    seg_states_str = seg_states_f.readline()
                    body_states_str = body_states_f.readline()
                    fil_states_str = fil_states_f.readline()

            FFwriter = animation.FFMpegWriter(fps=10)
            ani.save(f'fig/ciliate_{self.nfil}fil_anim.mp4', writer=FFwriter)
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i==self.plot_end_frame-1):
                    animation_func(i)
                else:
                    if show_flow_field:
                        seg_forces_str = seg_forces_f.readline()
                        blob_forces_str = blob_forces_f.readline()
                    seg_states_str = seg_states_f.readline()
                    body_states_str = body_states_f.readline()
                    fil_states_str = fil_states_f.readline()
            
            ax.set_aspect('equal')
            # plt.savefig(f'fig/ciliate_{self.nfil}fil_frame{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
            plt.savefig(f'fig/ciliate_{self.nfil}fil_frame{self.plot_end_frame}.png', bbox_inches = 'tight', format='png')
            plt.show()

    def ciliate_eco(self):        
        self.select_sim()

        # Fourier coeffs for the shape
        Ay = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
                    [4.0318e-01, -1.5553e+00, 7.3455e-01], \
                    [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
                    [8.1046e-02, -3.0982e-01, 1.4568e-01]])

        Ax = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
                    [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
                    [1.6209e-01, -3.4983e-01, 1.9082e-01], \
                    [1.0259e-02, 3.5907e-02, -6.8736e-02]])

        By = np.array([[0, 0, 0], \
                    [2.9136e-01, 1.0721e+00, -1.0433e+00], \
                    [6.1554e-03, 3.2521e-01, -2.8315e-01], \
                    [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

        Bx = np.array([[0, 0, 0], \
                    [1.9697e-01, -5.1193e-01, 3.4778e-01], \
                    [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
                    [1.2311e-02, 1.4157e-01, -1.1695e-01]])
        
            
        s_ref_filename = 'input/forcing/fulford_and_blake_reference_s_values_NSEG=20_SEP=2.600000.dat'

        fil_references_sphpolar = np.zeros((self.nfil,3))
        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])
        
        s_ref = np.loadtxt(s_ref_filename)
        num_ref_phase = s_ref[0]
        num_seg = int(s_ref[1])
        num_frame = 13
        num_points = 30
        radius = 1
        L = (num_seg-1)*2.6
        
        def fitted_shape_s(phase):
            cycle = 0.5*phase/np.pi*num_ref_phase
            sfloor = int(np.floor(cycle))
            sceil = sfloor + 1 if sfloor < 299 else 0

            floor_w = (cycle - sfloor)
            ceil_w = (sceil - cycle) 

            s = s_ref[2:][num_seg*sfloor:num_seg*sfloor+num_seg]*floor_w + s_ref[2:][num_seg*sceil:num_seg*sceil+num_seg]*ceil_w

            return s

        def fitted_shape(s, phase):
            pos = np.zeros(3)
            svec = np.array([s, s**2, s**3])
            fourier_dim = np.shape(Ax)[0]
            cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
            sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])
            cosvec[0] *= 0.5

            x = (cosvec@Ax + sinvec@Bx)@svec
            y = (cosvec@Ay + sinvec@By)@svec
            z = np.zeros(np.shape(x))

            return x, y, z

        seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        # Create the sphere data points
        num_points = 600
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)
        x = self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plotting
        # fig = plt.figure(dpi=600)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_proj_type('ortho')
        # ax.set_proj_type('persp', 0.05)  # FOV = 157.4 deg
        ax.view_init(elev=00., azim=90)
        ax.dist=5.8
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        global frame
        frame = 0

        def animation_func(t):
            global frame
            if(self.video):
                ax.cla()
            ax.axis('off')
            ax.set_aspect('equal')

            body_states_str = body_states_f.readline()
            seg_states_str = seg_states_f.readline()
            fil_states_str = fil_states_f.readline()

            body_states = np.array(body_states_str.split()[1:], dtype=float)
            seg_states = np.array(seg_states_str.split()[1:], dtype=float)
            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)
            
            fil_phases = fil_states[:self.nfil]
            fil_angles = fil_states[self.nfil:]
            
            for swim in range(self.nswim):
                # blob_data = np.zeros((int(self.pars['NBLOB']), 3))
                body_pos = body_states[7*swim : 7*swim+3]
                R = np.identity(3)
                R = util.rot_mat(body_states[7*swim+3 : 7*swim+7])
                
                # Plot the sphere
                if(t==self.plot_end_frame-1):
                    ax.plot_surface(x+body_pos[0], y+body_pos[1], z+body_pos[2], color='grey', alpha=0.5)
                body_axis_x = np.matmul(R, np.array([2*self.radius,0,0]))
                body_axis_y = np.matmul(R, np.array([0,2*self.radius,0]))
                body_axis_z = np.matmul(R, np.array([0,0,2*self.radius]))

                # Plot body axis
                # ax.plot([0, body_axis_x[0]]+body_pos[0], [0, body_axis_x[1]]+body_pos[1], [0, body_axis_x[2]]+body_pos[2])
                # ax.plot([0, body_axis_y[0]]+body_pos[0], [0, body_axis_y[1]]+body_pos[1], [0, body_axis_y[2]]+body_pos[2])
                # ax.plot([0, body_axis_z[0]]+body_pos[0], [0, body_axis_z[1]]+body_pos[1], [0, body_axis_z[2]]+body_pos[2])

                # Robot arm to find segment position (Ignored plane rotation!)
                for fil in range(self.nfil):
                    fil_base = body_pos + np.matmul(R, self.fil_references[3*fil : 3*fil+3])
                    fil_data = np.zeros((self.nseg, 3))

                    cmap_name = 'hsv'
                    # cmap_name = 'twilight_shifted'
                    cmap = plt.get_cmap(cmap_name)
                    fil_color = cmap(fil_phases[fil]/(2*np.pi))

                    s = np.linspace(0, 1, 20)
                    Rfil = util.rot_mat(self.fil_q[4*fil : 4*fil+4])

                    Rtheta = rotation_matrix = np.array([
                        [np.cos(fil_angles[fil]), -np.sin(fil_angles[fil]), 0],
                        [np.sin(fil_angles[fil]), np.cos(fil_angles[fil]), 0],
                        [0, 0, 1]
                    ])
                    for seg in range(0, int(self.pars['NSEG'])):
                        ref = self.fillength*R@Rfil@Rtheta@np.array(fitted_shape(s[seg], fil_phases[fil]))
                        seg_pos = fil_base + ref
                        fil_data[seg] = seg_pos

                    # Show only one side of the sphere
                    if fil_references_sphpolar[fil][1] > 0:
                        ax.plot(fil_data[:,0], fil_data[:,1], fil_data[:,2], c=fil_color, linewidth=3, zorder = 100)

        if(self.video):
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i>=self.plot_start_frame):
                    frame = i
                    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                    ani = animation.FuncAnimation(fig, animation_func, frames=self.frames, interval=10, repeat=False)
                    plt.show()
                    # FFwriter = animation.FFMpegWriter(fps=10)
                    # ani.save(f'fig/ciliate_{nfil}fil_anim.mp4', writer=FFwriter)
                else:
                    body_states_str = body_states_f.readline()
                    seg_states_str = seg_states_f.readline()
                    fil_states_str = fil_states_f.readline()
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                # if(i>self.plot_end_frame-30 and (i-self.plot_end_frame+1)%3 == 0):
                if(i==self.plot_end_frame-1):
                    animation_func(i)
                else:
                    body_states_str = body_states_f.readline()
                    seg_states_str = seg_states_f.readline()
                    fil_states_str = fil_states_f.readline()
                    frame += 1
            
            ax.set_aspect('equal')
            fig.tight_layout()
            # fig.savefig(f'fig/ciliate_index{self.index}_{self.date}_{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
            fig.savefig(f'fig/ciliate_index{self.index}_{self.date}_{self.plot_end_frame}.png', bbox_inches = 'tight', format='png', transparent=True)
            plt.show()

    def ciliate_traj(self):
        self.select_sim()

        body_states_f = open(self.simName + '_body_states.dat', "r")
        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )
        body_pos_array = np.zeros((len(time_array), 3))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            body_states_str = body_states_f.readline()

            if(i>=self.plot_start_frame):
                body_states = np.array(body_states_str.split()[1:], dtype=float)
                body_pos_array[i-self.plot_start_frame] = body_states[0 : 3]

        ax.plot(body_pos_array[:,0], body_pos_array[:,1], body_pos_array[:,2])
        plt.savefig(f'fig/ciliate_traj_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def ciliate_speed(self):
        self.select_sim()

        # seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        body_vels_f = open(self.simName + '_body_vels.dat', "r")

        # Plotting
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        
        # body_pos_array = np.zeros((len(time_array), 3))
        body_vel_array = np.zeros((len(time_array), 6))
        body_speed_array = np.zeros(len(time_array))

        # pos = np.zeros(3)

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            body_states_str = body_states_f.readline()
            body_vels_str = body_vels_f.readline()

            if(i>=self.plot_start_frame):

                body_states = np.array(body_states_str.split()[1:], dtype=float)
                body_vels = np.array(body_vels_str.split(), dtype=float)

                # body_pos_array[i-self.plot_start_frame] = body_states[0:3]
                body_vel_array[i-self.plot_start_frame] = body_vels
                body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))

            # pos += body_vels[0:3]
            # body_vel_array[i][0:3] = pos*self.dt

        # for i in range(len(body_vel_array[0])):
        #     # ax.plot(time_array, body_pos_array[:,i])

        

        def find_period(signal):
            min_diff = 1e10
            period = 1000
            for p in range(38, 45):
                diff = np.sum(np.abs(signal[:p] - signal[p:2*p]))
                if diff < min_diff:
                    min_diff = diff
                    period = p
            return period
    
        avg_speed = np.mean(body_speed_array)/self.fillength
        print(f'index={self.index} avg speed={avg_speed}')
        
        ax1.set_title(f'index={self.index} avg speed={avg_speed}')
        ax1.set_xlim(time_array[0], time_array[-1])
        ax1.plot(time_array, body_speed_array/self.fillength)
        ax1.set_ylabel(r"$|V|T/L$")
        ax1.set_xlabel(r"$t/T$")
        ax2.set_xlim(time_array[0], time_array[-1])
        ax2.plot(time_array, body_vel_array[:,2]/self.fillength)
        ax2.set_ylabel(r"$V_zT/L$")
        ax2.set_xlabel(r"$t/T$")

        plt.tight_layout()
        # fig1.savefig(f'fig/ciliate_speed_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        
        plt.show()

    def ciliate_speed_eco(self):
        self.select_sim()

        body_states_f = open(self.simName + '_body_states.dat', "r")

        # Plotting
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        
        body_pos_array = np.zeros((len(time_array), 3))
        body_axis_array = np.zeros((len(time_array), 3))
        body_q_array = np.zeros((len(time_array), 4))

        # body_vel_array = np.zeros((len(time_array), 6))
        # body_speed_array = np.zeros(len(time_array))


        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            body_states_str = body_states_f.readline()

            if(i>=self.plot_start_frame):

                body_states = np.array(body_states_str.split()[1:], dtype=float)

                body_pos_array[i-self.plot_start_frame] = body_states[0:3]
                R = util.rot_mat(body_states[3:7])
                body_axis_array[i-self.plot_start_frame] = np.matmul(R, np.array([0,0,1]))
                body_q_array[i-self.plot_start_frame] = body_states[3:7]


                # body_vel_array[i-self.plot_start_frame] = body_vels
                # body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
        
        body_vel_array = np.diff(body_pos_array, axis=0)/self.dt/self.fillength

        body_speed_array = np.linalg.norm(body_vel_array, axis=1)
        body_speed_along_axis_array = np.sum(body_vel_array * body_axis_array[:-1], axis=1)

        body_rot_vel_array = util.compute_angular_velocity(body_q_array, self.dt)
        body_rot_speed_along_axis_array = np.sum(body_rot_vel_array * body_axis_array[:-1], axis=1)
        body_rot_speed_array = np.linalg.norm(body_rot_vel_array, axis=1)

        avg_speed = np.mean(body_speed_along_axis_array)
        avg_rot_speed = np.mean(body_rot_speed_along_axis_array)
        print(f'index={self.index} avg speed={avg_speed} avg rot speed={avg_rot_speed}')

        ax1.plot(time_array[:-1], body_speed_along_axis_array)        
        ax1.set_title(f'index={self.index} avg speed={avg_speed}')
        ax1.set_xlim(time_array[0], time_array[-1])
        ax1.set_ylabel(r"$<V⋅e_1>/L$")
        ax1.set_xlabel(r"$t/T$")

        ax2.plot(time_array[:-1], body_rot_speed_along_axis_array)
        # ax2.set_xlim(time_array[0], time_array[-1])
        # ax2.plot(time_array, body_vel_array[:,2]/self.fillength)
        ax2.set_ylabel(r"$<Ω⋅e_1>$")
        ax2.set_xlabel(r"$t/T$")

        # np.save(f'{self.dir}/time_array_index{self.index}.npy', time_array)
        # np.save(f'{self.dir}/body_speed_array_index{self.index}.npy', body_speed_along_axis_array)
        # np.save(f'{self.dir}/body_rot_speed_array_index{self.index}.npy', body_rot_speed_along_axis_array)

        fig1.tight_layout()
        fig2.tight_layout()
        # fig1.savefig(f'fig/ciliate_speed_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/ciliate_rot_speed_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        
        plt.show()

    def ciliate_forcing(self):
        self.select_sim()

        seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
        blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
        

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )
        body_force_array = np.zeros((len(time_array), 3))

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            seg_forces_str = seg_forces_f.readline()
            blob_forces_str = blob_forces_f.readline()

            if(i>=self.plot_start_frame):

                seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)

                seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))

                body_force_array[i-self.plot_start_frame] = np.sum(blob_forces, axis=0) + np.sum(seg_forces[:,0:3], axis=0) 

        labels=[r'$\lambda_x$', r'$\lambda_y$', r'$\lambda_z$']
        for i in range(len(body_force_array[0])):
            ax.plot(time_array, body_force_array[:,i], label=labels[i])
        ax.set_xlabel('Time step')
        ax.set_ylabel('Force')
        plt.legend()
        plt.savefig(f'fig/ciliate_forcing_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def ciliate_dissipation(self):
        self.select_sim()

        seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
        seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
        blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
        blob_references_f = open(self.simName + '_blob_references.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        body_vels_f = open(self.simName + '_body_vels.dat', "r")

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        body_speed_array = np.zeros(self.frames)
        body_speed_along_axis_array = np.zeros(self.frames)
        body_rot_speed_array = np.zeros(self.frames)
        body_rot_speed_along_axis_array = np.zeros(self.frames)
        dissipation_array = np.zeros(self.frames)
        efficiency_array = np.zeros(self.frames)

        blob_references_str = blob_references_f.readline()
        blob_references= np.array(blob_references_str.split(), dtype=float)
        blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            seg_forces_str = seg_forces_f.readline()
            seg_vels_str = seg_vels_f.readline()
            blob_forces_str = blob_forces_f.readline()
            body_vels_str = body_vels_f.readline()
            body_states_str = body_states_f.readline()

            if(i>=self.plot_start_frame):
                seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                body_vels= np.array(body_vels_str.split()[1:], dtype=float)
                body_states = np.array(body_states_str.split()[1:], dtype=float)

                seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)
                
                R = util.rot_mat(body_states[3:7])
                body_axis = np.matmul(R, np.array([0,0,1]))
                

                body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                body_speed_along_axis_array[i-self.plot_start_frame] = np.sum(body_vels[0:3]*body_axis)

                
                body_rot_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[3:6]*body_vels[3:6], 0))
                body_rot_speed_along_axis_array[i-self.plot_start_frame] = np.sum(body_vels[3:6]*body_axis)

                dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)

        efficiency_array = 6*np.pi*self.radius*body_speed_along_axis_array**2/dissipation_array
        body_speed_along_axis_array /= self.fillength
        body_speed_array /= self.fillength
        dissipation_array /= self.fillength**3

        np.save(f'{self.dir}/time_array_index{self.index}.npy', time_array)
        np.save(f'{self.dir}/body_speed_array_index{self.index}.npy', body_speed_along_axis_array)
        np.save(f'{self.dir}/body_rot_speed_array_index{self.index}.npy', body_rot_speed_along_axis_array)
        np.save(f'{self.dir}/dissipation_array_index{self.index}.npy', dissipation_array)
        np.save(f'{self.dir}/efficiency_array_index{self.index}.npy', efficiency_array)

        ax.set_xlim(time_array[0], time_array[-1])
        ax.plot(time_array, dissipation_array)
        ax.set_xlabel(r'$t/T$')
        ax.set_ylabel(r'$PT^2/\mu L^3$')
        # fig.savefig(f'fig/ciliate_dissipation_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # plt.show()

    def ciliate_dmd(self):
        self.select_sim()
        
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        nfil = self.nfil
        n_snapshots = min(30, self.plot_end_frame)
        start = self.plot_end_frame - n_snapshots
        X = np.zeros((nfil, n_snapshots))
        r = min(nfil, n_snapshots-1)
        print(f'r={r}')
        dt = 1./30
        coeffs = np.zeros((r, n_snapshots), dtype=complex)

        fil_references_sphpolar = np.zeros((nfil,3))
        for fil in range(nfil):
            fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
        azim_array = fil_references_sphpolar[:,1]
        sorted_indices = np.argsort(azim_array)
        azim_array_sorted = azim_array[sorted_indices]

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_states_str = fil_states_f.readline()
            
            if(i>=start):
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                fil_phases = fil_states[:self.nfil]
                fil_phases = util.box(fil_phases, 2*np.pi)
                # fil_phases_sorted = fil_phases[sorted_indices]

                fil_phases_sorted = np.sin(fil_phases_sorted)

                X[:,i-start] = fil_phases_sorted[:nfil]
                
        X1 = X[:, :-1]
        X2 = X[:, 1:]

        U, sigma, V = np.linalg.svd(X1, full_matrices=False)
        Sigma = np.diag(sigma)
        V = V.conj().T
        U = U[:, :r]
        Sigma = Sigma[:r, :r]
        V = V[:, :r]


        A_tilde = U.conj().T @ X2 @ V @ np.linalg.inv(Sigma)
        D, W = np.linalg.eig(A_tilde)
        omega = np.log(D)/dt
        phi = X2 @ V @ np.linalg.inv(Sigma) @ W
        b = np.linalg.pinv(phi) @ X1[:,0]

        for t in range(n_snapshots):
            coeffs[:, t] = np.exp(omega * t*dt) * b

        X_dmd = (phi @ coeffs).real

            
        # print(np.shape(D), np.shape(W), np.shape(A_tilde))
        # print(np.shape(b))
        # print(np.shape(omega))
        # print(phi)
        # print(np.shape(np.exp(omega * dt)))

        inspected_snapshots = np.array([0, 1, 2])
        modes = np.abs(b).argsort()[-4:][::-1]
        # modes = [0,1,2,3]
    
        # Plotting
        fig, axs = plt.subplots(len(modes), sharex=True, sharey=True)
        axs_flat = axs.ravel()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)

        print(np.shape(phi))
        
        for ind, mode in enumerate(modes):
            axs[ind].plot(azim_array_sorted, phi[:, mode].real, label=f'real')
            axs[ind].plot(azim_array_sorted, phi[:, mode].imag, label=f'imag')

            # axs[ind].plot(coeffs[:, mode].real, label=f'real')
            # axs[ind].plot(coeffs[:, mode].imag, label=f'imag')

            # axs[ind].set_xlabel('Azimuth angle')
            # axs[ind].set_ylabel(r'$\phi$')
            axs[ind].set_title(f'mode={mode}')
            axs[ind].legend()
        axs[-1].set_xlabel("Azimuthal position")

        ax2.plot(np.abs(b))
        # for time in range(n_snapshots):
        #     ax2.plot(np.abs(coeffs[:, time]))
        ax2.set_xlabel(r'Mode')
        ax2.set_ylabel(r'Magnitude(b)')

        for i in inspected_snapshots:
            ax3.plot(X[:, i], c='b', marker='+')
            ax3.plot(X_dmd[:, i], c='r', marker='x')
        ax3.set_xlabel("fil")
        ax3.set_ylabel(r"sin(phase)")

        ax4.imshow(X)
        # ax4.plot(sigma)
        ax4.set_xlabel("t")
        ax4.set_ylabel(r"fil i")
        

        # fig.savefig(f'fig/fil_dmd_modes_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/fil_svd_cumsum_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        # fig3.savefig(f'fig/fil_svd_modes_{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def ciliate_svd(self):
        self.select_sim()
        spring_factor = self.pars_list['spring_factor'][0]
        
        fil_phases_f = open(self.simName + '_filament_phases.dat', "r")
        fil_angles_f = open(self.simName + '_filament_shape_rotation_angles.dat', "r")

        n_snapshots = min(12000, self.plot_end_frame)
        start = self.plot_end_frame - n_snapshots
        X = np.zeros((self.nfil, n_snapshots))
        X_angle = np.zeros((self.nfil, n_snapshots))

        fil_references_sphpolar = np.zeros((self.nfil,3))
        for fil in range(self.nfil):
            fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
        azim_array = fil_references_sphpolar[:,1]
        polar_array = fil_references_sphpolar[:,2]
        sorted_indices = np.argsort(azim_array)
        azim_array_sorted = azim_array[sorted_indices]
        polar_array_sorted = polar_array[sorted_indices]
        

        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            fil_phases_str = fil_phases_f.readline()
            fil_angles_str = fil_angles_f.readline()
            
            if(i>=start):
                fil_phases = np.array(fil_phases_str.split()[1:], dtype=float)
                fil_phases_sorted = fil_phases[sorted_indices]
                fil_phases_sorted = util.box(fil_phases_sorted, 2*np.pi)

                fil_phases_sorted = np.sin(fil_phases_sorted)

                fil_angles = np.array(fil_angles_str.split()[1:], dtype=float)
                fil_angles_sorted = fil_angles[sorted_indices]

                X[:,i-start] = fil_phases_sorted[:self.nfil]
                X_angle[:,i-start] = fil_angles_sorted[:self.nfil]

        U, sigma, V = np.linalg.svd(X, full_matrices=False)
        Sigma = np.diag(sigma)

        # np.savetxt(f'phase_data_20231201/spring_constant{spring_factor}/X_phase_index{self.index}.txt', X, delimiter=', ')
        # # np.savetxt(f'phase_data_20231107/by_index/spring_constant{spring_factor}/X_rotation_angle_index{self.index}.txt', X_angle, delimiter=', ')
        # np.savetxt(f'phase_data_20231201/spring_constant{spring_factor}/azim_pos_index{self.index}.txt', azim_array_sorted, delimiter=', ')
        # np.savetxt(f'phase_data_20231201/spring_constant{spring_factor}/polar_pos_index{self.index}.txt', polar_array_sorted, delimiter=', ')


        # pc = U @ Sigma
        # pa = V        
        
        # num_fil = 4
        # num_mode = 2
        # reduced = pc @ pa

        

        # # Plotting
        # colormap = 'twilight_shifted'
        # from matplotlib.colors import Normalize
        # from matplotlib.cm import ScalarMappable
        # norm = Normalize(vmin=0, vmax=2*np.pi)
        # sm = ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])

        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(1,1,1)
        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(1,1,1)
        # fig4 = plt.figure()
        # ax4 = fig4.add_subplot(1,1,1)
        # fig5 = plt.figure()
        # ax5 = fig5.add_subplot(1,1,1)
        # fig6 = plt.figure()
        # ax6 = fig6.add_subplot(1,1,1)
        # fig7 = plt.figure()
        # ax7 = fig7.add_subplot(1,1,1)
        # fig8 = plt.figure()
        # ax8 = fig8.add_subplot(1,1,1)

        # # Signal X
        # ax.imshow(X)
        # ax.set_xlabel('t')
        # ax.set_ylabel('Fil index (sorted by azimuth angle)')

        # # Signal of the first snapshot
        # ax2.scatter(azim_array_sorted, X[:,0])
        # ax2.set_xlabel('Azimuth position')
        # ax2.set_ylabel('Phase, t=0')

        # # Eigenvalue
        # ax3.scatter(np.arange(0, len(sigma), 1), sigma, marker='*')
        # ax3.set_xlabel(r'Mode')
        # ax3.set_ylabel(r'$Eigenvalue$')

        # # POD spatial modes
        # for i in range(2):
        #     ax4.plot(U[:,i], label=f'Mode {i}')
        #     ax4.set_xlabel('x')
        #     ax4.set_ylabel('f(x, t=0)')
        #     ax4.legend()

        # # POD temporal modes
        # for i in range(4):
        #     ax5.plot(V[i, :], label=f'Mode {i}')
        #     ax5.set_xlabel('t')
        #     ax5.set_ylabel('f(x, t=0)')
        #     ax5.legend()

        # # Phase of the first snapshot
        # ax6.scatter(azim_array_sorted, polar_array_sorted, c=X[:,0], cmap=colormap)
        # ax6.set_xlabel(r"Azimuth position")
        # ax6.set_ylabel(r"Polar position")

        # # Interpolated phase of the first snapshot
        # n1, n2 = 100, 100
        # azim_grid = np.linspace(-np.pi, np.pi, n1)
        # polar_grid = np.linspace(0, np.pi, n2)
        # xx, yy = np.meshgrid(azim_grid, polar_grid)
        # import scipy.interpolate
        # zz = scipy.interpolate.griddata((azim_array_sorted, polar_array_sorted), X[:,20], (xx, yy), method='cubic')
        # ax7.scatter(xx, yy, c=zz, cmap=colormap)
        # ax7.set_xlabel(r"Azimuth position")
        # ax7.set_ylabel(r"Polar position")
        
        # # Interpolated phase of reconstruction using first mode
        # uu = scipy.interpolate.griddata((azim_array_sorted, polar_array_sorted), U[:,1], (xx, yy), method='cubic')
        # ax8.scatter(xx, yy, c=uu, cmap=colormap)
        # ax8.set_xlabel(r"Azimuth position")
        # ax8.set_ylabel(r"Polar position")



        # # ax.scatter(azim_array_sorted, fil_phases_sorted)
        # # for fil in range(num_fil):
        # #     abs_pc = np.abs(pc[fil][:nfil])
        # #     ax2.plot(np.cumsum(abs_pc)/np.sum(abs_pc), label=f'fil {fil}')
        # # ax.set_xlabel('Azimuth angle')
        # # ax.set_ylabel(r'$\phi$')

        # # ax2.set_xlabel('Mode')
        # # ax2.set_ylabel('Accumulated |weight| fraction')
        # # ax2.legend()

        
        
        # # for i in range(6):
        # #     ax4.plot(U[i], c='b')
        # #     ax4.plot(reduced[i], c='r')

        # # ax4.plot(0, c='b', label='Original' )
        # # ax4.plot(0, c='r', label=f'Using {num_mode} modes')
        # # ax4.set_xlabel('Time step')
        # # ax4.set_ylabel(r'$\phi$')
        # # ax4.legend()
        

        # fig.savefig(f'fig/fil_svd_signal_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/fil_svd_initial_snapshot_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig3.savefig(f'fig/fil_svd_eigenvalues_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig4.savefig(f'fig/fil_svd_spatial_modes_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig5.savefig(f'fig/fil_svd_temporal_modes_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig6.savefig(f'fig/fil_svd_phase_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig7.savefig(f'fig/fil_svd_interpolated_phase_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # fig8.savefig(f'fig/fil_svd_first_mode_reconstruction_{self.nfil}fil_{self.index}.pdf', bbox_inches = 'tight', format='pdf')
        # plt.show()

    def timing(self):
        self.select_sim()

        with open(self.simName + '_time.dat', 'r') as file:
            plot_time_frame = len(file.readlines())
        timings_f = open(self.simName + '_time.dat', "r")

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        time_array = np.arange(0, plot_time_frame)
        
        time_end_frame = plot_time_frame
        time_start_frame = max(0, time_end_frame-100)
        timings_array = np.zeros((plot_time_frame, 7))
        total_timing_array = np.zeros(plot_time_frame)

        for i in range(plot_time_frame):
            print(" frame ", i, "/", plot_time_frame, "          ", end="\r")
            timings_str = timings_f.readline()

            if(i>=time_start_frame and time_end_frame):
                timings = np.array(timings_str.split(), dtype=float)
                timings_array[i] = timings[:7]
        labels = ['read_position',\
                  'assemble_rhs',\
                  'precondition',\
                  'mobility_mult',\
                  'k_mult',\
                  'gmres_solve',\
                  'finishing']
        for i in range(len(timings_array[0])):
            ax.plot(time_array, timings_array[:,i], label=labels[i])
            total_timing_array += timings_array[:,i]
        ax.plot(time_array, total_timing_array, label='Total')

        # ax.axvline(299, c='black')
        # plt.annotate('Double precision', (50, 0.055),fontsize=12)
        # plt.annotate('Single precision', (350, 0.035), fontsize=12)
        
        ax.set_ylabel("Computation time/s")
        ax.set_xlabel("Time step")
        ax.set_xlim(time_start_frame, time_end_frame)
        ax.set_ylim(0, 10)
        plt.legend()
        plt.savefig(f'fig/timings_{self.nfil}fil_{self.ar}ar.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def copy_phases(self):
        self.select_sim()

        input_filenames = [self.simName + '_filament_phases.dat',
                           self.simName + '_filament_shape_rotation_angles.dat',
                           self.simName + '_true_states.dat',
                           self.simName + '_body_states.dat']
        afix = int(self.index)
        output_filenames = [self.dir + f"phases{afix}.dat",
                            self.dir + f"angles{afix}.dat",
                            self.dir + f"psi{afix}.dat",
                            self.dir + f"bodystate{afix}.dat",
                            ]

        for i, name in enumerate(input_filenames):
            input_filename = name
            output_filename = output_filenames[i]
            try:
                # Open the input file in read mode
                with open(input_filename, 'r') as input_file:
                    # Read all lines from the file
                    lines = input_file.readlines()

                    # Check if the file is not empty
                    if lines:
                        # Extract the last line
                        first_line = lines[0]
                        last_line = lines[-1]

                        data = np.array(last_line.split()[1:], dtype=float)
                        # data[:self.nfil] = util.box(data[:self.nfil], 2*np.pi)

                        if name == self.simName + '_true_states.dat':
                            data = np.concatenate(([self.spring_factor], data))

                        np.savetxt(output_filename, data, delimiter=' ', newline=' ')

                        print(f"[SUCCESS]: last line copied from '{input_filename}' to '{output_filename}'.")
                    else:
                        print(f"The file '{input_filename}' is empty.")
            except FileNotFoundError:
                print(f"Error: The file '{input_filename}' does not exist.")

        # fil_phases = np.array(combine_text[0].split()[1:], dtype=float)
        # fil_phases = util.box(fil_phases, 2*np.pi)
        # fil_angles = np.array(combine_text[1].split()[1:], dtype=float)
        # combined_par = np.concatenate((fil_phases, fil_angles))
        # np.savetxt(self.dir + f"psi{int(self.index)}.dat", combined_par, delimiter=' ', newline=' ')       

    def find_periodicity(self):
        self.select_sim()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)

        fil_references_sphpolar = np.zeros((self.nfil,3))
        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])

        near_pole_ind = np.where(np.sin(fil_references_sphpolar[:,2]) < 0.0 )            

        # fil_phases_f = open(self.simName + '_filament_phases.dat', "r")
        # fil_angles_f = open(self.simName + '_filament_shape_rotation_angles.dat', "r")
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        states = np.zeros((self.frames, 2*self.nfil))
        pi_diff = np.zeros(2*self.nfil)
        pi_diff[:self.nfil] = 2*np.pi


        # Read and store states
        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
            # fil_phases_str = fil_phases_f.readline()
            # fil_angles_str = fil_angles_f.readline()
            fil_states_str = fil_states_f.readline()

            if(i>=self.plot_start_frame):
                # states[i-self.plot_start_frame][:self.nfil] = np.array(fil_phases_str.split()[1:], dtype=float)
                # states[i-self.plot_start_frame][self.nfil:] = np.array(fil_angles_str.split()[1:], dtype=float)
                states[i-self.plot_start_frame] = np.array(fil_states_str.split()[2:], dtype=float)
        
        # Function to compute diff
        def compute_diff(arr, frame1, frame2):
            aux = np.copy(arr[frame1]) # using the F(x) norm instead of F(x+T)
            aux[:self.nfil] = util.box(aux[:self.nfil], 2*np.pi)
            aux[:self.nfil] = np.ones(self.nfil)*np.pi
            
            diff = arr[frame2] - arr[frame1] - pi_diff

            # Excluding fils near the pole
            for ind in near_pole_ind:
                diff[ind] = 0.
                diff[ind+self.nfil] = 0.

            diff_norm = float(np.linalg.norm(diff)) 
            norm = float(np.linalg.norm(aux))
            rel_error = diff_norm / norm
            return diff_norm, norm, rel_error

        # Search for T
        search_T_min, search_T_max = 0.96, 1.0
        dframe_array = np.arange(int(search_T_min*self.period), int(search_T_max*self.period)+1)
        error_array = np.zeros(np.shape(dframe_array))
        try:
            for ti, dframe in enumerate(dframe_array):
                # scan_range = np.arange(dframe, self.frames, dtype=int)
                scan_range = [-1]
                for frame in scan_range:
                    error_array[ti] += compute_diff(states, frame-dframe, frame)[2]

                error_array[ti] /= len(scan_range)
        except:
            print(f"No. of frames {self.frames} not enough to find the period")

        ax.plot(dframe_array*self.dt, error_array, marker = '+')
        min_ind = int(np.where(error_array==error_array.min())[0])
        dframe_soln = dframe_array [min_ind]
        T_soln = dframe_soln * self.dt
        print(f'\033[32mThe real period is T= {T_soln}({dframe_soln} frames)\
               \nwith rel error |x({self.plot_end_frame-1})-x({self.plot_end_frame-1-dframe_soln})|/|x({self.plot_end_frame-1-dframe_soln})|={error_array.min()}\033[m')
        
        # # Create interpolated states
        # from scipy.interpolate import CubicSpline
        # m, n = states.shape
        # interpolation_factor = 2.4
        # p = int( (self.frames - 1) * interpolation_factor +1 )
        # new_indices = np.linspace(0, self.frames - 1, self.frames)
        # interpolated_arrays = []
        # for i in range(n):
        #     cubic_spline = CubicSpline(new_indices, states[:, i])
        #     interpolated_dimension = cubic_spline(np.linspace(0, m - 1, p))
        #     interpolated_arrays.append(interpolated_dimension)
        # states_interpolated = np.column_stack(interpolated_arrays)
        
        # # Search for T using cubic spline
        # dframe_array = np.arange(int(search_T_min*self.period*interpolation_factor), int(search_T_max*self.period*interpolation_factor)+1)
        # error_array = np.zeros(np.shape(dframe_array))
        # try:
        #     for ti, dframe in enumerate(dframe_array):
        #         scan_range = [-1]
        #         for frame in scan_range:
        #             error_array[ti] += compute_diff(states_interpolated, frame-dframe, frame)[2]

        #         error_array[ti] /= len(scan_range)
        # except:
        #     print("No. of frames not enough to find the period")
        # dt = self.dt/interpolation_factor
        # ax.plot(dframe_array*dt, error_array, marker = '+')


        # Calculate the continuous period
        # def linfit(x, k, c):
        #     return k*x + c
        # popt, pcov = curve_fit(linfit, dframe_array[min_ind-5:min_ind]*self.dt, error_array[min_ind-5:min_ind])
        # ax.plot(dframe_array*self.dt, linfit(dframe_array*self.dt, popt[0], popt[1]))

        # popt2, pcov2 = curve_fit(linfit, dframe_array[min_ind+1:min_ind+6]*self.dt, error_array[min_ind+1:min_ind+6])
        # ax.scatter(dframe_array[min_ind+1:min_ind+6]*self.dt, error_array[min_ind+1:min_ind+6])
        # ax.plot(dframe_array*self.dt, linfit(dframe_array*self.dt, popt2[0], popt2[1]))
        

        # Calculate the error |F(x) - x| using the computed T.
        possible_length = int(min(self.period, self.frames-1))
        error_of_final_period = compute_diff(states, -1-int(possible_length), -1)[2]

        print(f'\033[33mThe beat period is T= {self.dt*self.period}({int(self.period)} frames)\
               \nwith rel error |x({self.plot_end_frame-1})-x({self.plot_end_frame-1-possible_length})|/|x({self.plot_end_frame-1-possible_length})|={error_of_final_period}\033[m')


        # plot error over time for the computed period
        ts = np.arange(self.plot_start_frame+dframe_soln, self.plot_end_frame)/self.period
        diff_norms = np.zeros(self.frames-dframe_soln)
        norms = np.zeros(self.frames-dframe_soln)
        phase_avgs = np.zeros(self.frames-dframe_soln)
        angle_avgs = np.zeros(self.frames-dframe_soln)
        diff_avgs = np.zeros(self.frames-dframe_soln)
        rel_error = np.zeros(self.frames-dframe_soln)
        order_parameters = np.zeros(self.frames-dframe_soln)
        for i in range(self.frames-dframe_soln):
            diff_norms[i], norms[i], rel_error[i] = compute_diff(states, i, dframe_soln+i )

        ax2.plot(ts, rel_error)

        ax3.plot(ts, norms, label='Norm of states')
        ax3_right = ax3.twinx()
        ax3_right.plot(ts, diff_norms, c='r', label=r'Norm of diff')

        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"<$\frac{\|\psi(t_0+T)-\psi(t_0)\|}{\|\psi(t_0)\|}$>")
        ax.set_ylim(0, 0.02)
        ax.set_xlim(0.9, 1.)
        fig.tight_layout()

        # ax2.set_ylim(0, 1e-2)
        ax2.set_xlabel(r"$t/T$")
        ax2.set_ylabel(r"<$\frac{\|\psi(t_0+T)-\psi(t_0)\|}{\|\psi(t_0)\|}$>")
        fig2.tight_layout()

        ax3.set_xlabel(r"$t/T$")
        ax3.set_ylabel(r"$<\|\psi(t)\|>$")
        ax3_right.set_ylabel(r"$<\|\psi(t_0+T)-\psi(t_0)\|>$")
        ax3.legend(loc='upper left')
        ax3_right.legend(loc=1)
        # ax3_right.set_ylim(0, 1e-2)
        fig3.tight_layout()

        fig.savefig(f'fig/fil_finding_period_index{self.index}_{self.date}.pdf', bbox_inches = 'tight', format='pdf')    
        fig2.savefig(f'fig/fil_period_errod_index{self.index}_{self.date}.pdf', bbox_inches = 'tight', format='pdf')    
        plt.show()

    def periodic_solution(self):

        input_filename = self.dir + f"psi_guess.dat"

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        try:
            # Open the input file in read mode
            with open(input_filename, 'r') as input_file:
                # Read all lines from the file
                lines = np.loadtxt(input_filename)
                
                k_list = lines[:,0]
                T_list = lines[:,1]
                ax.plot(k_list, T_list, marker='+')
                ax.set_xlabel(r'$k$')
                ax.set_ylabel(r'$T$')
                fig.savefig(f'fig/T_solution.pdf', bbox_inches = 'tight', format='pdf')
        
        except FileNotFoundError:
            print(f"Error: The file '{input_filename}' does not exist.")
        
        plt.show()

    def flow_field_2D(self):
        
        read_flow_field = False

        def stokeslet(x, x0, f0):
            r = np.linalg.norm(x-x0)
            if r == 0:
                return np.zeros(3)
            return f0/(8.*np.pi*r) + np.dot(f0, x-x0) * (x - x0) / (8.*np.pi*r**3)

        def compute_v_list(pos_list, source_pos_list, source_force_list, v_list):
            for pi, pos in enumerate(pos_list):
                print(" pi ", pi, "/", len(pos_list), "          ", end="\r")
                for si, source_pos in enumerate(source_pos_list):
                    source_force = source_force_list[si]
                    v_list[pi] += stokeslet(pos, source_pos, source_force)

        @cuda.jit(device=True)
        def stokeslet_device(x, x0, f0, result):
            dis_x = x[0] - x0[0]
            dis_y = x[1] - x0[1]
            dis_z = x[2] - x0[2]
            r = (dis_x**2 + dis_y**2 + dis_z**2)**0.5
            
            coeff = 1 / (8. * np.pi * r)
            dot_product = f0[0] * dis_x + f0[1] * dis_y + f0[2] * dis_z
            vector_term_x = dot_product * dis_x / (8. * np.pi * r**3)
            vector_term_y = dot_product * dis_y / (8. * np.pi * r**3)
            vector_term_z = dot_product * dis_z / (8. * np.pi * r**3)
            
            result[0] = coeff*f0[0] + vector_term_x
            result[1] = coeff*f0[1] + vector_term_y
            result[2] = coeff*f0[2] + vector_term_z

        @cuda.jit
        def compute_v_list_kernel(pos_list, source_pos_list, source_force_list, v_list):
            pi = cuda.grid(1)
            if pi < pos_list.shape[0]:
                pos = pos_list[pi]
                v_temp = cuda.local.array(3, float64)  # Local array for temporary storage
                v_temp[0] = 0.0
                v_temp[1] = 0.0
                v_temp[2] = 0.0
                for si in range(source_pos_list.shape[0]):
                    source_pos = source_pos_list[si]
                    source_force = source_force_list[si]
                    temp_result = cuda.local.array(3, dtype=float64)
                    stokeslet_device(pos, source_pos, source_force, temp_result)
                    v_temp[0] += temp_result[0]
                    v_temp[1] += temp_result[1]
                    v_temp[2] += temp_result[2]
                v_list[pi, 0] += v_temp[0]
                v_list[pi, 1] += v_temp[1]
                v_list[pi, 2] += v_temp[2]

        # need to test
        self.select_sim()

        seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
        blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
        seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")
        fil_states_f = open(self.simName + '_true_states.dat', "r")

        fig = plt.figure()
        ax = fig.add_subplot()
        cmap_name = 'hsv'
        cmap = plt.get_cmap(cmap_name)
        
        # # plane
        # n_field_point = 12
        # y_lower, y_upper = 128-50, 128+51
        # z_lower, z_upper = 3, 74
        # x_list = [126]
        # y_list = np.arange(y_lower, y_upper, 5)
        # z_list = np.arange(z_lower, z_upper, 5)

        # sphere 
        r_ratio = 5.
        x_lower, x_upper = -r_ratio*self.radius, r_ratio*self.radius, 
        y_lower, y_upper = -r_ratio*self.radius, r_ratio*self.radius, 
        z_lower, z_upper = -r_ratio*self.radius, r_ratio*self.radius, 

        flow_spacing = 20
        speed_limit = 40

        # x_list = np.arange(y_lower, y_upper+0.01, flow_spacing)
        # y_list = np.arange(y_lower, y_upper+0.01, flow_spacing)
        # z_list = np.arange(z_lower, z_upper+0.01, flow_spacing)

        # x_mesh, y_mesh, z_mesh = np.meshgrid(x_list, y_list, z_list)
        # x_mesh = x_mesh.swapaxes(0, 1)
        # y_mesh = y_mesh.swapaxes(0, 1)
        # z_mesh = z_mesh.swapaxes(0, 1)

        x_mesh, y_mesh, z_mesh = np.mgrid[x_lower:x_upper:flow_spacing, y_lower:y_upper:flow_spacing, z_lower:z_upper:flow_spacing ]

        print(x_mesh.shape)


        x_flat = x_mesh.flatten()
        y_flat = y_mesh.flatten()
        z_flat = z_mesh.flatten()

        pos_list = np.column_stack((x_flat, y_flat, z_flat))

        r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        theta_flat = np.arctan2(y_flat, x_flat)
        phi_flat = np.arccos(z_flat / r_flat)

        global frame
        frame = 0

        def animation_func(t):
            global frame
            print(frame)
            ax.cla()
            ax.set_xlim(y_lower, y_upper)
            ax.set_ylim(z_lower, z_upper)

            seg_forces_str = seg_forces_f.readline()
            blob_forces_str = blob_forces_f.readline()
            seg_states_str = seg_states_f.readline()
            body_states_str = body_states_f.readline()
            fil_states_str = fil_states_f.readline()

            seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
            blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
            seg_states = np.array(seg_states_str.split()[1:], dtype=float)
            body_states = np.array(body_states_str.split()[1:], dtype=float)
            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)
            fil_phases = fil_states[:self.nfil]

            fil_plot_data = np.zeros((self.nfil, self.nseg, 3))

            if read_flow_field:
                v_list = np.load(f'{self.dir}/v_list_index{self.index}_{frame}.npy')
            else:
                v_list = np.zeros(np.shape(pos_list))

            # Create arrays to store pos and force of all particles
            source_pos_list = np.zeros((self.nfil*self.nseg + self.nblob, 3))
            source_force_list = np.zeros((self.nfil*self.nseg + self.nblob, 3))

            for swim in range(int(self.pars['NSWIM'])):
                body_pos = body_states[7*swim : 7*swim+3]
                circle=plt.Circle((body_pos[1], body_pos[2]), self.radius, color='Grey', zorder=99)
                ax.add_patch(circle)

                for blob in range(self.nblob):
                    blob_pos = np.array(util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3]))
                    blob_force = blob_forces[3*blob : 3*blob+3]
                    # ax.scatter(blob_pos[1], blob_pos[2], c='black')

                    source_pos_list[blob] = blob_pos
                    source_force_list[blob] = blob_force

                for fil in range(self.nfil):
                    fil_i = int(3*fil*self.nseg)
                    seg_data = np.zeros((self.nseg, 3))
                    fil_color = cmap(fil_phases[fil]/(2*np.pi))
                    for seg in range(self.nseg):
                        seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)]
                        seg_data[seg] = seg_pos
                        seg_force = seg_forces[2*fil_i+6*(seg) : 2*fil_i+6*(seg+1)]
                        seg_force = seg_force[:3]

                        source_pos_list[self.nblob+fil*self.nseg+seg] = seg_pos
                        source_force_list[self.nblob+fil*self.nseg+seg] = seg_force

                    fil_plot_data[fil] = seg_data
                    # only plot fil when the fil is facing us. this is done by checking the base of the filament
                    if(seg_data[0, 0]>0):
                        # ax.plot(seg_data[:,0], seg_data[:,1], c=fil_color, zorder = 100)
                        ax.plot(seg_data[:,1], seg_data[:,2], c=fil_color, zorder = 100)

            # compute the flow field
            if not read_flow_field:
                # using single core
                # compute_v_list(pos_list, source_pos_list, source_force_list, v_list)

                # using GPU
                # Move data to GPU
                d_pos_list = cuda.to_device(pos_list)
                d_source_pos_list = cuda.to_device(source_pos_list)
                d_source_force_list = cuda.to_device(source_force_list)
                d_v_list = cuda.to_device(v_list)
                # Define the grid and block dimensions
                threads_per_block = 256
                blocks_per_grid = (pos_list.shape[0] + threads_per_block - 1) // threads_per_block
                # Launch the kernel
                compute_v_list_kernel[blocks_per_grid, threads_per_block](d_pos_list, d_source_pos_list, d_source_force_list, d_v_list)
                # Copy the result back to the host
                v_list = d_v_list.copy_to_host()


                ur_list = v_list[:, 0] * np.sin(theta_flat) * np.cos(phi_flat) + \
                            v_list[:, 1] * np.sin(theta_flat) * np.sin(phi_flat) + \
                                v_list[:, 2] * np.cos(theta_flat)
                uphi_list = - v_list[:, 0] * np.sin(phi_flat) + v_list[:, 1] * np.cos(phi_flat)
                utheta_list = v_list[:, 0] * np.cos(theta_flat) * np.cos(phi_flat) + \
                                v_list[:, 1] * np.cos(theta_flat) * np.sin(phi_flat) \
                                    - v_list[:, 2] * np.sin(theta_flat)

            # Flow field
            cmap_name2= 'seismic'
            speed_list = np.linalg.norm(v_list, axis=1)
            max_speed = max(speed_list)
            avg_speed = np.mean(speed_list)
            np.clip(speed_list, None, speed_limit, speed_list)

            print(f'maxspeed={max_speed}  avgspeed={avg_speed}')

            speed_mesh = speed_list.reshape(y_mesh.shape)
            ur_mesh = ur_list.reshape(y_mesh.shape)

            half_plane_index = int(x_mesh.shape[0]/2)
            speed_mesh_2D = speed_mesh[half_plane_index,:,:]

            vx_mesh = v_list[:,0].reshape(y_mesh.shape)
            vy_mesh = v_list[:,1].reshape(y_mesh.shape)
            vz_mesh = v_list[:,2].reshape(y_mesh.shape)

            x_mesh_2D = x_mesh[half_plane_index,:,:]
            y_mesh_2D = y_mesh[half_plane_index,:,:]
            z_mesh_2D = z_mesh[half_plane_index,:,:]
            vx_mesh_2D = vx_mesh[half_plane_index,:,:]
            vy_mesh_2D = vy_mesh[half_plane_index,:,:]
            vz_mesh_2D = vz_mesh[half_plane_index,:,:]


            # phi_var_plot = ax.imshow(ur_mesh, cmap='jet', origin='lower', extent=[y_lower, y_upper, z_lower, z_upper], vmax = 20, vmin=-20)
            phi_var_plot = ax.imshow(speed_mesh_2D.T, cmap=cmap_name2, origin='lower', extent=[y_lower, y_upper, z_lower, z_upper], vmax = speed_limit, vmin=0)
            
            
            # Colorbars
            # from matplotlib.colors import Normalize
            # from matplotlib.cm import ScalarMappable
            # norm = Normalize(vmin=0, vmax=2*np.pi)
            # sm = ScalarMappable(cmap=cmap_name, norm=norm)
            # sm.set_array([])
            # cbar = plt.colorbar(sm)
            # cbar.ax.set_yticks(np.linspace(0, 2*np.pi, 7), ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
            # cbar.set_label(r"Phase")
            # cbar2 = plt.colorbar(phi_var_plot) 
            # cbar2.set_label(r"|$v$|")

            # ax.scatter(pos_list[:,1], pos_list[:,2], color=colors)
            # ax.quiver(pos_list[:,1], pos_list[:,2], normalised_v_list[:,1], normalised_v_list[:,2], scale_units='xy',scale=3.)
            
            # ax.streamplot(y_mesh_2D.T, z_mesh_2D.T, vy_mesh_2D.T, vz_mesh_2D.T, color='black', density=0.5, broken_streamlines=False)
            ax.streamplot(y_mesh_2D.T, z_mesh_2D.T, vy_mesh_2D.T, vz_mesh_2D.T, color='black', density=0.5, broken_streamlines=False)

            np.save(f'{self.dir}/vx_mesh_index{self.index}_{frame}.npy', vx_mesh)
            np.save(f'{self.dir}/vy_mesh_index{self.index}_{frame}.npy', vy_mesh)
            np.save(f'{self.dir}/vz_mesh_index{self.index}_{frame}.npy', vz_mesh)
            np.save(f'{self.dir}/x_mesh_index{self.index}_{frame}.npy', x_mesh)
            np.save(f'{self.dir}/y_mesh_index{self.index}_{frame}.npy', y_mesh)
            np.save(f'{self.dir}/z_mesh_index{self.index}_{frame}.npy', z_mesh)
            np.save(f'{self.dir}/fil_data_index{self.index}_{frame}.npy', fil_plot_data)
            np.save(f'{self.dir}/fil_phases_index{self.index}_{frame}.npy', fil_phases)

            frame += 1

        if(self.video):
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i>=self.plot_start_frame):
                    frame = i
                    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                    ani = animation.FuncAnimation(fig, animation_func, frames=self.frames, interval=10, repeat=False)
                    # plt.show()
                    break
                else:
                    seg_forces_str = seg_forces_f.readline()
                    blob_forces_str = blob_forces_f.readline()
                    seg_states_str = seg_states_f.readline()
                    body_states_str = body_states_f.readline()
                    fil_states_str = fil_states_f.readline()

            FFwriter = animation.FFMpegWriter(fps=10)
            ani.save(f'fig/flowfield_2D_{self.nfil}fil_anim.mp4', writer=FFwriter)
        else:
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                if(i==self.plot_end_frame-1):
                    frame = i
                    animation_func(i)
                else:
                    seg_forces_str = seg_forces_f.readline()
                    blob_forces_str = blob_forces_f.readline()
                    seg_states_str = seg_states_f.readline()
                    body_states_str = body_states_f.readline()
                    fil_states_str = fil_states_f.readline()
            
            ax.set_aspect('equal')
            plt.savefig(f'fig/flowfield2D_{self.nfil}fil_frame{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
            plt.show()

    def flow_field_kymograph(self):
        def stokeslet(x, x0, f0):
            r = np.linalg.norm(x-x0)
            if r == 0:
                return np.zeros(3)
            return f0/(8.*np.pi*r) + np.dot(f0, x-x0) * (x - x0) / (8.*np.pi*r**3)

        def compute_v_list(pos_list, source_pos_list, source_force_list, v_list):
            for pi, pos in enumerate(pos_list):
                print(" pi ", pi, "/", len(pos_list), "          ", end="\r")
                for si, source_pos in enumerate(source_pos_list):
                    source_force = source_force_list[si]
                    v_list[pi] += stokeslet(pos, source_pos, source_force)

        @cuda.jit(device=True)
        def stokeslet_device(x, x0, f0, result):
            dis_x = x[0] - x0[0]
            dis_y = x[1] - x0[1]
            dis_z = x[2] - x0[2]
            r = (dis_x**2 + dis_y**2 + dis_z**2)**0.5
            
            coeff = 1 / (8. * np.pi * r)
            dot_product = f0[0] * dis_x + f0[1] * dis_y + f0[2] * dis_z
            vector_term_x = dot_product * dis_x / (8. * np.pi * r**3)
            vector_term_y = dot_product * dis_y / (8. * np.pi * r**3)
            vector_term_z = dot_product * dis_z / (8. * np.pi * r**3)
            
            result[0] = coeff*f0[0] + vector_term_x
            result[1] = coeff*f0[1] + vector_term_y
            result[2] = coeff*f0[2] + vector_term_z

        @cuda.jit
        def compute_v_list_kernel(pos_list, source_pos_list, source_force_list, v_list):
            pi = cuda.grid(1)
            if pi < pos_list.shape[0]:
                pos = pos_list[pi]
                v_temp = cuda.local.array(3, float64)  # Local array for temporary storage
                v_temp[0] = 0.0
                v_temp[1] = 0.0
                v_temp[2] = 0.0
                for si in range(source_pos_list.shape[0]):
                    source_pos = source_pos_list[si]
                    source_force = source_force_list[si]
                    temp_result = cuda.local.array(3, dtype=float64)
                    stokeslet_device(pos, source_pos, source_force, temp_result)
                    v_temp[0] += temp_result[0]
                    v_temp[1] += temp_result[1]
                    v_temp[2] += temp_result[2]
                v_list[pi, 0] += v_temp[0]
                v_list[pi, 1] += v_temp[1]
                v_list[pi, 2] += v_temp[2]

        # need to test
        self.select_sim()

        seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
        blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
        seg_states_f = open(self.simName + '_seg_states.dat', "r")
        body_states_f = open(self.simName + '_body_states.dat', "r")

        fig = plt.figure()
        ax = fig.add_subplot()

        # Flow field
        n_theta = 30
        n_r = 1
        n_phi = 10
        n_field_point = n_theta*n_r*n_phi

        r_ratio = 1.5
        r_list = np.linspace(r_ratio, 2.6, n_r)*self.radius
        theta_list = np.linspace(0, np.pi, n_theta)
        phi_list = np.linspace(0, 2*np.pi, n_phi+1)[:-1]

        R, Phi, Theta = np.meshgrid(r_list, phi_list, theta_list, indexing='ij')

        r_flat = R.ravel()
        theta_flat = Theta.ravel()
        phi_flat = Phi.ravel()

        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)

        x_flat = X.ravel()
        y_flat = Y.ravel()
        z_flat = Z.ravel()

        pos_list = np.column_stack((x_flat, y_flat, z_flat))

        v_data = np.zeros((self.frames, n_field_point, 3))
        ur_data = np.zeros((self.frames, n_field_point))
        utheta_data = np.zeros((self.frames, n_field_point))
        uphi_data = np.zeros((self.frames, n_field_point))        
        
        for i in range(self.plot_end_frame):
            print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")

            seg_forces_str = seg_forces_f.readline()
            blob_forces_str = blob_forces_f.readline()
            seg_states_str = seg_states_f.readline()
            body_states_str = body_states_f.readline()

            if(i>=self.plot_start_frame):
                seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                seg_states = np.array(seg_states_str.split()[1:], dtype=float)
                body_states = np.array(body_states_str.split()[1:], dtype=float)

                source_pos_list = np.zeros((self.nfil*self.nseg + self.nblob, 3))
                source_force_list = np.zeros((self.nfil*self.nseg + self.nblob, 3))
                v_list = np.zeros(np.shape(pos_list))

                
                for swim in range(int(self.pars['NSWIM'])):
                    for blob in range(self.nblob):
                        blob_pos = np.array(util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3]))
                        blob_force = blob_forces[3*blob : 3*blob+3]
                        source_pos_list[blob] = blob_pos
                        source_force_list[blob] = blob_force
                    for fil in range(self.nfil):
                        fil_i = int(3*fil*self.nseg)
                        seg_data = np.zeros((self.nseg, 3))
                        for seg in range(self.nseg):
                            seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)]
                            seg_data[seg] = seg_pos
                            seg_force = seg_forces[2*fil_i+6*(seg) : 2*fil_i+6*(seg+1)]
                            seg_force = seg_force[:3]
                            source_pos_list[self.nblob+fil*self.nseg+seg] = seg_pos
                            source_force_list[self.nblob+fil*self.nseg+seg] = seg_force
                
                d_pos_list = cuda.to_device(pos_list)
                d_source_pos_list = cuda.to_device(source_pos_list)
                d_source_force_list = cuda.to_device(source_force_list)
                d_v_list = cuda.to_device(v_list)
                # Define the grid and block dimensions
                threads_per_block = 256
                blocks_per_grid = (pos_list.shape[0] + threads_per_block - 1) // threads_per_block
                # Launch the kernel
                compute_v_list_kernel[blocks_per_grid, threads_per_block](d_pos_list, d_source_pos_list, d_source_force_list, d_v_list)
                # Copy the result back to the host
                v_list = d_v_list.copy_to_host()


                # for swim in range(int(self.pars['NSWIM'])):
                #     for blob in range(int(self.pars['NBLOB'])):
                #         # print(" blob ", blob, "          ", end="\r")
                #         blob_pos = np.array(util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3]))
                #         blob_force = blob_forces[3*blob : 3*blob+3]
                #         for pi, pos in enumerate(pos_list):
                #             v_list[pi] += stokeslet(pos, blob_pos, blob_force)
                #     for fil in range(int(self.pars['NFIL'])):
                #         # print(" fil ", fil, "          ", end="\r")
                #         fil_i = int(3*fil*self.pars['NSEG'])
                #         for seg in range(int(self.pars['NSEG'])):
                #             seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)]
                #             seg_force = seg_forces[2*fil_i+6*(seg) : 2*fil_i+6*(seg+1)]
                #             seg_force = seg_force[:3]
                #             for pi, pos in enumerate(pos_list):
                #                 v_list[pi] += stokeslet(pos, seg_pos, seg_force)

                v_data[i-self.plot_start_frame] = v_list

                ur_list = v_list[:, 0] * np.sin(theta_flat) * np.cos(phi_flat) + \
                                v_list[:, 1] * np.sin(theta_flat) * np.sin(phi_flat) + \
                                    v_list[:, 2] * np.cos(theta_flat)
                uphi_list = - v_list[:, 0] * np.sin(phi_flat) + v_list[:, 1] * np.cos(phi_flat)
                utheta_list = v_list[:, 0] * np.cos(theta_flat) * np.cos(phi_flat) + \
                                v_list[:, 1] * np.cos(theta_flat) * np.sin(phi_flat) \
                                    - v_list[:, 2] * np.sin(theta_flat)

                ur_data[i-self.plot_start_frame] = ur_list
                utheta_data[i-self.plot_start_frame] = utheta_list
                uphi_data[i-self.plot_start_frame] = uphi_list

                np.save(f'{self.dir}/ur_data_fil{self.nfil}_r{r_ratio}.npy', ur_data)
                np.save(f'{self.dir}/utheta_data_fil{self.nfil}_r{r_ratio}.npy', utheta_data)
                np.save(f'{self.dir}/grid_shape_fil{self.nfil}_r{r_ratio}.npy', np.array(R.shape))            

        t = ur_data.shape[0]/30
        
        # ax.imshow(ur_data.T, cmap='jet', origin='upper', extent=[0, t, 0, 2*np.pi])

        # y_ticks = np.linspace(0, 2*np.pi, 5)
        # y_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$' ][::-1]
        # ax.set_yticks(ticks=y_ticks, labels=y_labels)

        # ax.set_xlabel(r'$t/T$')
        # ax.set_ylabel(r'$\theta$')        

        # # fig.savefig(f'fig/flow_field{with_blob_string}_frame{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
        # # fig.savefig(f'fig/flow_field{with_blob_string}_frame{self.plot_end_frame}.png', bbox_inches = 'tight', format='png')
        # plt.show()
    
# Multi sims
    def multi_phase(self):
        # Plotting
        colormap = 'cividis'
        colormap = 'twilight_shifted'

        # nrow = len(np.unique(self.pars_list['nfil']))
        # ncol = len(np.unique(self.pars_list['ar']))
        # if(ncol == 1 or nrow == 1):
            # nrow = int(self.num_sim**.5)
            # ncol = nrow + (1 if nrow**2 < self.num_sim else 0)
        
        ncol = self.ncol
        nrow = self.num_sim//ncol

        print(f'num sim = {self.num_sim} nrow = {nrow} ncol = {ncol}')
        spring_factor = self.pars_list['spring_factor'][0]

        
        fig, axs = plt.subplots(nrow, ncol, figsize=(18, 18), sharex=True, sharey=True)

        axs_flat = axs.ravel()
        # Row-major, bottom-to-top indices
        row_indices = np.arange(nrow - 1, -1, -1)
        col_indices = np.arange(ncol)
        row_major_bottom_top_indices = np.array([
            row + nrow * col for row in row_indices for col in col_indices
        ])
        print(row_major_bottom_top_indices)


        import scipy.interpolate

        # fig.supxlabel(r"Azimuth position")
        # fig.supylabel(r"Polar position")
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        # plt.ylabel(r"$\theta$")
        # plt.xlabel(r"$\phi$")
        
        plt.xlim(-np.pi, np.pi)
        plt.ylim(0, np.pi)
        plt.xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
        plt.yticks(np.linspace(0, np.pi, 3), ['0', 'π/2', 'π'])
        plt.gca().invert_yaxis()

        for ind, ax in zip(row_major_bottom_top_indices, axs_flat):
        # for ind, ax in enumerate(axs_flat):
            ax.invert_yaxis()
            if (ind < self.num_sim):
                try:
                    self.index = ind
                    self.select_sim()

                    fil_references_sphpolar = np.zeros((self.nfil,3))
                    
                    fil_states_f = open(self.simName + '_true_states.dat', "r")
                    # fil_phases_f = open(self.simName + '_filament_phases.dat', "r")
                    # fil_angles_f = open(self.simName + '_filament_shape_rotation_angles.dat', "r")
                    for i in range(self.plot_end_frame):
                        print(" index ", self.index,  " frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                        fil_states_str = fil_states_f.readline()
                        # fil_phases_str = fil_phases_f.readline()
                        # fil_angles_str = fil_angles_f.readline()
                        if(i==self.plot_end_frame-1):
                            # fil_phases = np.array(fil_phases_str.split()[1:], dtype=float)
                            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                            fil_phases = fil_states[:self.nfil]
                            fil_phases = util.box(fil_phases, 2*np.pi)
                            for i in range(self.nfil):
                                fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])
                            
                            cmap = mpl.colormaps[colormap]
                            colors = cmap(fil_phases/2/np.pi)
                            if (self.interpolate):
                                n1, n2 = 64, 64
                                offset = 0.2
                                azim_grid = np.linspace(min(fil_references_sphpolar[:,1])+offset, max(fil_references_sphpolar[:,1])-offset, n1)
                                polar_grid = np.linspace(min(fil_references_sphpolar[:,2])+offset, max(fil_references_sphpolar[:,2])-offset, n2)
                                xx, yy = np.meshgrid(azim_grid, polar_grid)
                                xx, yy = xx.ravel(), yy.ravel()

                                
                                colors_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), colors, (xx, yy), method='linear')
                                ax.scatter(xx, yy, c=colors_inter)

                            else:
                            # Individual filaments
                                ax.scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)

                    # ax.set_title(f"ind={self.index} spr={self.spring_factor} {self.plot_end_frame}")
                    ax.set_title(f"ind={self.plot_end_frame}")
                except:
                    print("WARNING: " + self.simName + " not found.")
        for ax in axs_flat:
            ax.tick_params(axis='both', which='both', labelsize=18)
        plt.tight_layout()
        # plt.savefig(f'fig/ciliate_multi_phase_elst{spring_factor}.png', bbox_inches = 'tight', format='png')
        plt.savefig(f'fig/ciliate_multi_phase_{self.date}_{self.plot_end_frame}.pdf', bbox_inches = 'tight', format='pdf')
        # plt.show()

    def multi_kymograph(self):
        # Plotting
        colormap = 'cividis'
        colormap = 'twilight_shifted'
        
        ncol = self.ncol
        nrow = self.num_sim//ncol

        print(f'num sim = {self.num_sim} nrow = {nrow} ncol = {ncol}')
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(18, 18), sharex=True, sharey=True)
        # fig2, axs2 = plt.subplots(nrow, ncol, figsize=(18, 18), sharex=True, sharey=True)
        # cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height] for the colorbar

        axs_flat = axs.ravel()
        # axs2_flat = axs2.ravel()
        import scipy.interpolate

        # fig.supxlabel(r"Azimuth position")
        # fig.supylabel(r"Polar position")
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        # plt.ylabel(r"$\theta$")
        # plt.xlabel(r"$\phi$")

        
        # plt.ylim(0, np.pi)
        # plt.xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
        plt.yticks(np.linspace(0, np.pi, 3), ['0', 'π/2', 'π'])
        # plt.gca().invert_yaxis()

        for ind, ax in enumerate(axs_flat):
            if (ind < self.num_sim):
                try:
                    self.index = ind
                    self.select_sim()

                    ##
                    time_array = np.arange(self.frames)/self.period
                    n1, n2 = 100, 50 
                    azim_array = np.linspace(-np.pi, np.pi, n1)
                    polar_array = np.linspace(0, np.pi, n2)

                    phi_kymo = np.zeros((n1, self.frames))
                    theta_kymo = np.zeros((n2, self.frames))
                    
                    phi_kymo_xx, phi_kymo_yy = np.meshgrid(time_array, azim_array,)
                    theta_kymo_xx, theta_kymo_yy = np.meshgrid(time_array, polar_array, )
                    ##

                    fil_references_sphpolar = np.zeros((self.nfil,3))
                    for m in range(self.nfil):
                        fil_references_sphpolar[m] = util.cartesian_to_spherical(self.fil_references[3*m: 3*m+3])
        
                    fil_phases_f = open(self.simName + '_filament_phases.dat', "r")
                    # fil_angles_f = open(self.simName + '_filament_shape_rotation_angles.dat', "r")
                    for i in range(self.plot_end_frame):
                        print(" index ", self.index,  " frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                        fil_phases_str = fil_phases_f.readline()
                        # fil_angles_str = fil_angles_f.readline()
                        if(i>=self.plot_start_frame):
                            fil_phases = np.array(fil_phases_str.split()[1:], dtype=float)
                            fil_phases = util.box(fil_phases, 2*np.pi)

                            xx, yy = np.meshgrid(azim_array, polar_array)
                            zz = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), fil_phases, (xx, yy), method='nearest')

                            phi_kymo[:, i-self.plot_start_frame] = zz[n2//2]
                            theta_kymo[:, i-self.plot_start_frame] = zz[:, n1//2]

                    # ax.scatter(phi_kymo_xx, phi_kymo_yy, c=phi_kymo, cmap=colormap, vmin=0, vmax=2*np.pi)
                    ax.scatter(theta_kymo_xx, theta_kymo_yy, c=theta_kymo, cmap=colormap, vmin=0, vmax=2*np.pi)       
            
                    ax.set_title(f"ind={self.index} nfil={self.nfil} AR={self.ar} spr={self.spring_factor} {self.plot_end_frame}")
                except:
                    print("WARNING: " + self.simName + " not found.")
        for ax in axs_flat:
            ax.tick_params(axis='both', which='both', labelsize=18)
        plt.xlim(time_array[0], time_array[-1])
        plt.tight_layout()
        
        plt.savefig(f'fig/ciliate_multi_kymograph_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def multi_ciliate(self):
        # Plotting
        nrow = int(self.num_sim**.5)
        ncol = nrow + (1 if nrow**2 < self.num_sim else 0)
        fig, axs = plt.subplots(nrow, ncol, figsize=(18, 18), subplot_kw={'projection': '3d'})
        axs_flat = axs.ravel()

        for ind, ax in enumerate(axs_flat):
            ax.set_proj_type('ortho')
            # ax.set_proj_type('persp', 0.05)  # FOV = 157.4 deg
            # ax.view_init(elev=5., azim=45)
            # ax.dist=20
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # ax.axis('off')
            # ax.grid(False)

            # ax.set_xlim(-1000, 1000)
            # ax.set_ylim(-1000, 1000)
            # ax.set_zlim(-1000, 1000)
            if (ind < self.num_sim):
                try:
                    self.index = ind
                    self.select_sim()

                    # Create the sphere data points
                    num_points = 300
                    u = np.linspace(0, 2 * np.pi, num_points)
                    v = np.linspace(0, np.pi, num_points)
                    x = self.radius * np.outer(np.cos(u), np.sin(v))
                    y = self.radius * np.outer(np.sin(u), np.sin(v))
                    z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v))

                    seg_states_f = open(self.simName + '_seg_states.dat', "r")
                    body_states_f = open(self.simName + '_body_states.dat', "r")

                    for i in range(self.plot_end_frame):
                        print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                        body_states_str = body_states_f.readline()
                        seg_states_str = seg_states_f.readline()

                        if(i==self.plot_end_frame-1):
                            body_states = np.array(body_states_str.split()[1:], dtype=float)
                            seg_states = np.array(seg_states_str.split()[1:], dtype=float)

                            for swim in range(self.nswim):
                                body_pos = body_states[7*swim : 7*swim+3]
                                if(self.big_sphere):
                                    ax.plot_surface(x+body_pos[0], y+body_pos[1], z+body_pos[2], color='grey', alpha=0.5)
                                    # ax.plot_surface(x, y, z, color='grey', alpha=0.5)
                                
                                else:
                                    R = util.rot_mat(body_states[7*swim+3 : 7*swim+7])
                                    for blob in range(int(self.pars['NBLOB'])):
                                        blob_x, blob_y, blob_z = util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3])
                                        ax.scatter(blob_x, blob_y, blob_z)
                                
                                # Robot arm to find segment position (Ignored plane rotation!)
                                for fil in range(self.nfil):
                                    fil_data = np.zeros((self.nseg, 3))
                                    fil_i = int(3*fil*self.nseg)
                                    fil_data[0] = seg_states[fil_i : fil_i+3]

                                    for seg in range(1, self.nseg):
                                        seg_pos = seg_states[fil_i+3*(seg-1) : fil_i+3*seg]
                                        fil_data[seg] = seg_pos
                                    ax.plot(fil_data[:,0], fil_data[:,1], fil_data[:,2], c='black', zorder = 100)
                    ax.set_title(f"nfil={self.nfil} AR={self.ar}")
                except:
                    print("WARNING: " + self.simName + " not found.")

        plt.tight_layout()
        plt.savefig(f'fig/ciliate_multi.png', bbox_inches = 'tight', format='png')
        plt.savefig(f'fig/ciliate_multi.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def multi_ciliate_traj(self):
        # Plotting
        nrow = int(self.num_sim**.5)
        ncol = nrow + (1 if nrow**2 < self.num_sim else 0)
        fig, axs = plt.subplots(nrow, ncol, figsize=(18, 18), subplot_kw={'projection': '3d'})
        axs_flat = axs.ravel()

        for ind, ax in enumerate(axs_flat):
            ax.set_proj_type('ortho')
            if (ind < self.num_sim):
                try:
                    self.index = ind
                    self.select_sim()

                    body_states_f = open(self.simName + '_body_states.dat', "r")
                    time_array = np.arange(self.plot_start_frame, self.plot_end_frame )
                    body_pos_array = np.zeros((self.frames, 3))

                    for i in range(self.plot_end_frame):
                        print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                        body_states_str = body_states_f.readline()

                        if(i>=self.plot_start_frame):
                            body_states = np.array(body_states_str.split()[1:], dtype=float)

                            body_pos_array[i-self.plot_start_frame] = body_states[0 : 3]

                    ax.plot(body_pos_array[:,0], body_pos_array[:,1], body_pos_array[:,2])
                    ax.set_title(f"nfil={self.nfil} AR={self.ar}")
                except:
                    print("WARNING: " + self.simName + " not found.")

        plt.tight_layout()
        plt.savefig(f'fig/ciliate_multi_traj.png', bbox_inches = 'tight', format='png')
        plt.savefig(f'fig/ciliate_multi_traj.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def multi_ciliate_speed(self):
         # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        avg_speed = np.zeros(self.num_sim)
        max_speed = np.zeros(self.num_sim)
        some_speed = np.zeros(self.num_sim)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                body_vels_f = open(self.simName + '_body_vels.dat', "r")

                time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        
                # body_vel_array = np.zeros((self.frames, 6))
                body_speed_array = np.zeros(self.frames)

                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    body_vels_str = body_vels_f.readline()

                    if(i>=self.plot_start_frame):
                        body_vels = np.array(body_vels_str.split(), dtype=float)

                        # body_vel_array[i-self.plot_start_frame] = body_vels
                        body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))

                avg_speed[ind] = np.mean(body_speed_array)
                max_speed[ind] = np.max(body_speed_array)
                some_speed[ind] = body_speed_array[0]
                ax.plot(time_array, body_speed_array/self.fillength, label=f"{self.index}) nblob={self.nblob}")
            except:
                print("WARNING: " + self.simName + " not found.")

        print(f'avg speed={avg_speed/self.fillength}')

        print(", ".join((max_speed/self.fillength).astype(str)))
        print(", ".join((some_speed/self.fillength).astype(str)))
        # print(f'max speed={max_speed/self.fillength}')
        
        plt.legend()
        ax.set_xlabel(r'$t/T$')
        ax.set_ylabel(r'$<VT/L>$')
        ax.set_xlim(0, 1)
        fig.savefig(f'fig/multi_speed_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def multi_ciliate_dissipation(self):
         # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)

        max_speed = np.zeros(self.num_sim)
        some_speed = np.zeros(self.num_sim)
        max_dissipation = np.zeros(self.num_sim)
        some_dissipation = np.zeros(self.num_sim)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")

                time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
                body_speed_array = np.zeros(self.frames)
                dissipation_array = np.zeros(self.frames)
                efficiency_array = np.zeros(self.frames)

                blob_references_str = blob_references_f.readline()
                blob_references= np.array(blob_references_str.split(), dtype=float)
                blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))

                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    seg_forces_str = seg_forces_f.readline()
                    seg_vels_str = seg_vels_f.readline()
                    blob_forces_str = blob_forces_f.readline()
                    body_vels_str = body_vels_f.readline()

                    if(i>=self.plot_start_frame):
                        seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                        seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                        blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                        body_vels= np.array(body_vels_str.split(), dtype=float)

                        seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                        body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                        blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                        body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                        dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)

                efficiency_array = 6*np.pi*self.radius*body_speed_array**2/dissipation_array
                ax.plot(time_array, body_speed_array/self.fillength, label=f"index={self.index}")
                ax2.plot(time_array, dissipation_array/self.fillength**3, label=f"index={self.index}")
                ax3.plot(time_array, efficiency_array, label=f"index={self.index}")
                
                max_speed[ind] = np.max(body_speed_array)
                some_speed[ind] = body_speed_array[0]
                max_dissipation[ind] = np.max(dissipation_array)
                some_dissipation[ind] = dissipation_array[0]
            except:
                print("WARNING: " + self.simName + " not found.")
        
        print(", ".join((max_speed/self.fillength).astype(str)))
        print(", ".join((some_speed/self.fillength).astype(str)))
        print(", ".join((max_dissipation/self.fillength**3).astype(str)))
        print(", ".join((some_dissipation/self.fillength**3).astype(str)))

        ax.set_xlabel(r'$t/T$')
        ax.set_ylabel(r'$VT/L$')
        ax2.set_xlabel(r'$t/T$')
        ax2.set_ylabel(r'$PT^2/\mu L^3$')
        ax3.set_xlabel(r'$t/T$')
        ax3.set_ylabel(r'Efficiency')

        fig.legend()
        fig2.legend()
        fig3.legend()
        fig.savefig(f'fig/multi_speed_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig(f'fig/multi_dissipation_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig3.savefig(f'fig/multi_efficiency_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def multi_ciliate_dissipation_generate(self):
         # Plotting
        for ind in range(self.num_sim):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1,1,1)
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(1,1,1)
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(1,1,1)
            try:
                self.index = ind
                self.select_sim()

                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")

                time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        
                body_speed_array = np.zeros(self.frames)
                dissipation_array = np.zeros(self.frames)
                efficiency_array = np.zeros(self.frames)

                blob_references_str = blob_references_f.readline()
                blob_references= np.array(blob_references_str.split(), dtype=float)
                blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))

                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    seg_forces_str = seg_forces_f.readline()
                    seg_vels_str = seg_vels_f.readline()
                    blob_forces_str = blob_forces_f.readline()
                    body_vels_str = body_vels_f.readline()

                    if(i>=self.plot_start_frame):
                        seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                        seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                        blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                        body_vels= np.array(body_vels_str.split(), dtype=float)

                        seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                        body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                        blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                        body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                        dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)

                # efficiency_array = 6*np.pi*self.radius*body_speed_array**2/dissipation_array
                # ax.plot(time_array, body_speed_array/self.fillength, label=f"index={self.index}")
                # ax2.plot(time_array, dissipation_array/self.fillength**3, label=f"index={self.index}")
                # ax3.plot(time_array, efficiency_array, label=f"index={self.index}")
                # ax4.plot(time_array, dissipation_array/self.fillength**3/self.nfil, label=f"index={self.index}")

                np.savetxt(f'other/data_over_time/speed_{self.date}_index{self.index}.txt', body_speed_array, delimiter=', ')
                np.savetxt(f'other/data_over_time/dissipation_{self.date}_index{self.index}.txt', dissipation_array, delimiter=', ')

                # ax.set_xlabel(r'$t/T$')
                # ax.set_ylabel(r'$VT/L$')
                # ax.set_xlim(time_array[0], time_array[-1])
                # ax2.set_xlabel(r'$t/T$')
                # ax2.set_ylabel(r'$PT^2/\mu L^3$')
                # ax2.set_xlim(time_array[0], time_array[-1])
                # ax3.set_xlabel(r'$t/T$')
                # ax3.set_ylabel(r'Efficiency')
                # ax3.set_xlim(time_array[0], time_array[-1])
                # ax4.set_xlabel(r'$t/T$')
                # ax4.set_ylabel(r'$<PT^2/\mu L^3 N_{fil}>$')
                # ax4.set_xlim(time_array[0], time_array[-1])

                # fig.legend(loc=0)
                # fig2.legend(loc=0)
                # fig3.legend(loc=0)
                # fig4.legend(loc=0)
                # fig.savefig(f'fig/data_over_time/speed_{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
                # fig2.savefig(f'fig/data_over_time/dissipation_{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
                # fig3.savefig(f'fig/data_over_time/efficiency_{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
                # fig4.savefig(f'fig/data_over_time/dissipation_per_cilium{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
                
                # plt.show()
                
            except:
                print("WARNING: " + self.simName + " not found.")

    def multi_ciliate_dissipation_plots(self):
        # Plotting
        for ind in range(self.num_sim):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1,1,1)
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(1,1,1)
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(1,1,1)

            self.index = ind
            self.select_sim()

            speed_array_f = open(f'other/data_over_time/speed_{self.date}_index{self.index}.txt', "r")
            dissipation_array_f = open(f'other/data_over_time/dissipation_{self.date}_index{self.index}.txt', "r")

            self.plot_end_frame = min(self.plot_end_frame_setting, sum(1 for line in open(f'other/data_over_time/speed_{self.date}_index{self.index}.txt')))
            self.plot_start_frame = max(0, self.plot_end_frame-self.frames_setting)
            self.frames = self.plot_end_frame - self.plot_start_frame

            time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
            body_speed_array = np.array([float(line.strip()) for line in speed_array_f])[self.plot_start_frame:self.plot_end_frame]
            dissipation_array = np.array([float(line.strip()) for line in dissipation_array_f])[self.plot_start_frame:self.plot_end_frame]
            efficiency_array = 6*np.pi*self.radius*body_speed_array**2/dissipation_array

            ax.plot(time_array, body_speed_array/self.fillength, label=f"index={self.index}")
            ax2.plot(time_array, dissipation_array/self.fillength**3, label=f"index={self.index}")
            ax3.plot(time_array, efficiency_array, label=f"index={self.index}")
            ax4.plot(time_array, dissipation_array/self.fillength**3/self.nfil, label=f"index={self.index}")

            ax.set_xlabel(r'$t/T$')
            ax.set_ylabel(r'$<VT/L>$')
            ax.set_xlim(time_array[0], time_array[-1])
            ax2.set_xlabel(r'$t/T$')
            ax2.set_ylabel(r'$<P>$')
            ax2.set_xlim(time_array[0], time_array[-1])
            ax3.set_xlabel(r'$t/T$')
            ax3.set_ylabel(r'<Efficiency>')
            ax3.set_xlim(time_array[0], time_array[-1])
            ax4.set_xlabel(r'$t/T$')
            ax4.set_ylabel(r'$<P/N>$')
            ax4.set_xlim(time_array[0], time_array[-1])

            fig.legend()
            fig2.legend()
            fig3.legend()
            fig4.legend()
            fig.savefig(f'fig/data_over_time/speed_{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
            fig2.savefig(f'fig/data_over_time/dissipation_{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
            fig3.savefig(f'fig/data_over_time/efficiency_{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
            fig4.savefig(f'fig/data_over_time/dissipation_per_cilium{self.date}_index{self.index}.pdf', bbox_inches = 'tight', format='pdf')
            
            # plt.show()

    def multi_ciliate_special_func(self):
        dates = ['20231105', '20231111']
        num_sim = np.array([16, 24])
        total_num_sim = np.sum(num_sim)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)

        nfil_list = np.zeros(total_num_sim)
        density_list = np.zeros(total_num_sim)
        ar_list = np.zeros(total_num_sim)
        fillen_list = np.zeros(total_num_sim)
        sphere_r_list = np.zeros(total_num_sim)
        speed_list = np.zeros(total_num_sim)
        angular_speed_list = np.zeros(total_num_sim)
        dissipation_list = np.zeros(total_num_sim)
        efficiency_list = np.zeros(total_num_sim)
        from_which_sim_list = np.zeros(total_num_sim)

        ncol = 4
        
        for di, date in enumerate(dates):
            self.date = date
            self.dir = f"data/expr_sims/{self.date}/"
            self.read_rules()

            # Plotting
            for ind in range(num_sim[di]):
                start_ind = di*num_sim[0]
                self.index = ind
                self.select_sim()

                speed_array_f = open(f'other/data_over_time/speed_{self.date}_index{self.index}.txt', "r")
                dissipation_array_f = open(f'other/data_over_time/dissipation_{self.date}_index{self.index}.txt', "r")

                self.plot_end_frame = min(self.plot_end_frame_setting, sum(1 for line in open(f'other/data_over_time/speed_{self.date}_index{self.index}.txt')))
                self.plot_start_frame = max(0, self.plot_end_frame-self.frames_setting)
                self.frames = self.plot_end_frame - self.plot_start_frame
                print(self.plot_start_frame, self.plot_end_frame)

                body_speed_array = np.array([float(line.strip()) for line in speed_array_f])[self.plot_start_frame:self.plot_end_frame]
                dissipation_array = np.array([float(line.strip()) for line in dissipation_array_f])[self.plot_start_frame:self.plot_end_frame]
                efficiency_array = 6*np.pi*self.radius*body_speed_array**2/dissipation_array

                nfil_list[ind+start_ind] = self.nfil
                speed_list[ind+start_ind] = np.mean(body_speed_array)
                dissipation_list[ind+start_ind] = np.mean(dissipation_array)
                efficiency_list[ind+start_ind] = np.mean(efficiency_array)
                sphere_r_list[ind+start_ind] = self.radius
                ar_list[ind+start_ind] = self.ar
                fillen_list[ind+start_ind] = self.fillength
                density_list[ind+start_ind] = self.fildensity
                if di == 0:
                    from_which_sim_list[ind+start_ind] = 1

        linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', '' ]
        for i in range(ncol):
            plot_fil_list = nfil_list[i::ncol]
            sorted_indices = np.argsort(plot_fil_list)
            plot_fil_list = plot_fil_list[sorted_indices]
            plot_speed_list = speed_list[i::ncol][sorted_indices]
            plot_fillen_list = fillen_list[i::ncol][sorted_indices]
            plot_dis_list = dissipation_list[i::ncol][sorted_indices]
            plot_eff_list = efficiency_list[i::ncol][sorted_indices]
            plot_from_which_sim_list = from_which_sim_list[i::ncol][sorted_indices]
            color_list = list()
            for l in range(len(plot_from_which_sim_list)):
                if plot_from_which_sim_list[l] == 1:
                    color_list.append('black')
                else:
                    color_list.append('black')

            ax.scatter(plot_fil_list, plot_speed_list/plot_fillen_list, marker='+', c=color_list)
            ax2.scatter(plot_fil_list, plot_dis_list/plot_fillen_list **3, marker='+', c=color_list)
            ax3.scatter(plot_fil_list, plot_eff_list, marker='+', c=color_list)
            ax4.scatter(plot_fil_list, plot_dis_list/plot_fil_list/plot_fillen_list **3, marker='+', c=color_list)

            ax.plot(plot_fil_list, plot_speed_list/plot_fillen_list, linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2f}")
            ax2.plot(plot_fil_list, plot_dis_list/plot_fillen_list **3, linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2f}")
            ax3.plot(plot_fil_list, plot_eff_list, linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2f}")
            ax4.plot(plot_fil_list, plot_dis_list/plot_fil_list/plot_fillen_list **3, linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2f}")

        ax.legend()
        # ax2.legend()
        ax3.legend()
        ax4.legend(loc='upper left')
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$<VT/L>$')
        ax2.set_xlabel(r'$N$')
        ax2.set_ylabel(r'$<P>$')
        ax3.set_xlabel(r'$N$')
        ax3.set_ylabel(r'$<Efficiency>$')
        ax4.set_xlabel(r'$N$')
        ax4.set_ylabel(r'$<P/N>$')
        ax4.set_ylim(5.3, 8.0)

        fig.savefig(f'fig/speed_vs_nfil.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig(f'fig/dissipation_vs_nfil.pdf', bbox_inches = 'tight', format='pdf')
        fig3.savefig(f'fig/efficiency_vs_nfil.pdf', bbox_inches = 'tight', format='pdf')
        fig4.savefig(f'fig/dissipation_per_cilium_vs_nfil.pdf', bbox_inches = 'tight', format='pdf')
        
        plt.show()

    def multi_ciliate_special_func2(self):
        dates = ['20231105', '20231111']
        num_sim = np.array([16, 44])
        total_num_sim = np.sum(num_sim)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)

        nfil_list = np.zeros(total_num_sim)
        density_list = np.zeros(total_num_sim)
        ar_list = np.zeros(total_num_sim)
        fillen_list = np.zeros(total_num_sim)
        sphere_r_list = np.zeros(total_num_sim)
        speed_list = np.zeros(total_num_sim)
        angular_speed_list = np.zeros(total_num_sim)
        dissipation_list = np.zeros(total_num_sim)
        efficiency_list = np.zeros(total_num_sim)
        from_which_sim_list = np.zeros(total_num_sim)

        ncol = 4
        nrow = int(total_num_sim/4)
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 
        
        for di, date in enumerate(dates):
            self.date = date
            self.dir = f"data/expr_sims/{self.date}/"
            self.read_rules()

            # Plotting
            for ind in range(num_sim[di]):
                start_ind = di*num_sim[0]
                self.index = ind
                self.select_sim()

                speed_array_f = open(f'other/data_over_time/speed_{self.date}_index{self.index}.txt', "r")
                dissipation_array_f = open(f'other/data_over_time/dissipation_{self.date}_index{self.index}.txt', "r")

                self.plot_end_frame = min(self.plot_end_frame_setting, sum(1 for line in open(f'other/data_over_time/speed_{self.date}_index{self.index}.txt')))
                self.plot_start_frame = max(0, self.plot_end_frame-self.frames_setting)
                self.frames = self.plot_end_frame - self.plot_start_frame
                print(self.plot_start_frame, self.plot_end_frame)

                body_speed_array = np.array([float(line.strip()) for line in speed_array_f])[self.plot_start_frame:self.plot_end_frame]
                dissipation_array = np.array([float(line.strip()) for line in dissipation_array_f])[self.plot_start_frame:self.plot_end_frame]
                efficiency_array = 6*np.pi*self.radius*body_speed_array**2/dissipation_array

                nfil_list[ind+start_ind] = self.nfil
                speed_list[ind+start_ind] = np.mean(body_speed_array)
                dissipation_list[ind+start_ind] = np.mean(dissipation_array)
                efficiency_list[ind+start_ind] = np.mean(efficiency_array)
                sphere_r_list[ind+start_ind] = self.radius
                ar_list[ind+start_ind] = self.ar
                fillen_list[ind+start_ind] = self.fillength
                density_list[ind+start_ind] = self.fildensity
                if di == 0:
                    from_which_sim_list[ind+start_ind] = 1

        # linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'densely dotted', 
        #                  'long dash with offset', 'loosely dashed', 'loosely dashdotted', 'dashdotted',
        #                  'dashdotdotted', 'densely dashed', 'densely dashdotdotted', 'loosely dashdotdotted']
        
        for i in range(nrow):
            plot_density_list = density_list[4*i:4*i+4]

            plot_fil_list = nfil_list[4*i:4*i+4]
            print(plot_fil_list)
            plot_speed_list = speed_list[4*i:4*i+4]
            plot_fillen_list = fillen_list[4*i:4*i+4]
            plot_dis_list = dissipation_list[4*i:4*i+4]
            plot_eff_list = efficiency_list[4*i:4*i+4]
            plot_from_which_sim_list = from_which_sim_list[4*i:4*i+4]
            color_list = list()
            for l in range(len(plot_from_which_sim_list)):
                if plot_from_which_sim_list[l] == 1:
                    color_list.append('r')
                else:
                    color_list.append('black')

            ax.scatter(plot_density_list, plot_speed_list/plot_fillen_list, marker='+', c=color_list)
            ax2.scatter(plot_density_list, plot_dis_list/plot_fillen_list **3, marker='+', c=color_list)
            ax3.scatter(plot_density_list, plot_eff_list, marker='+', c=color_list)
            ax4.scatter(plot_density_list, plot_dis_list/plot_fil_list/plot_fillen_list **3, marker='+', c=color_list)

            ax.plot(plot_density_list, plot_speed_list/plot_fillen_list, label=r"$N_{fil}=$"+f"{plot_fil_list[0]:.0f}")
            ax2.plot(plot_density_list, plot_dis_list/plot_fillen_list **3, label=r"$N_{fil}=$"+f"{plot_fil_list[0]:.0f}")
            ax3.plot(plot_density_list, plot_eff_list, label=r"$N_{fil}=$"+f"{plot_fil_list[0]:.0f}")
            ax4.plot(plot_density_list, plot_dis_list/plot_fil_list/plot_fillen_list **3, label=r"$N_{fil}=$"+f"{plot_fil_list[0]:.0f}")

        ax.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax.set_xlabel(r'$N_{fil}/4\pi r^2$')
        ax.set_ylabel(r'$<VT/L>$')
        ax2.set_xlabel(r'$N_{fil}/4\pi r^2$')
        ax2.set_ylabel(r'$<PT^2/\mu L^3>$')
        ax3.set_xlabel(r'$N_{fil}/4\pi r^2$')
        ax3.set_ylabel(r'$<Efficiency>$')
        ax4.set_xlabel(r'$N_{fil}/4\pi r^2$')
        ax4.set_ylabel(r'$<PT^2/\mu L^3 N_{fil}>$')

        fig.savefig(f'fig/speed_vs_density.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig(f'fig/dissipation_vs_density.pdf', bbox_inches = 'tight', format='pdf')
        fig3.savefig(f'fig/efficiency_vs_density.pdf', bbox_inches = 'tight', format='pdf')
        fig4.savefig(f'fig/dissipation_per_cilium_vs_density.pdf', bbox_inches = 'tight', format='pdf')
        
        plt.show()      

    def multi_order_parameter(self):
         # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure(figsize=(20, 12),)
        ax3 = fig3.add_subplot(1,1,1)

        print(self.pars_list['spring_factor'])

        colormap = 'tab20'
        cmap = mpl.colormaps[colormap]

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                fil_states_f = open(self.simName + '_true_states.dat', "r")
                # fil_phases_f = open(self.simName + '_filament_phases.dat', "r")

                time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
                corr_array = np.zeros(self.frames)
                corr_array2 = np.zeros(self.frames)
                corr_array_angle = np.zeros(self.frames)
                r_array = np.zeros(self.frames)

                fil_references_sphpolar = np.zeros((self.nfil,3))
                for fil in range(self.nfil):
                    fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
                azim_array = fil_references_sphpolar[:,1]
                polar_array = fil_references_sphpolar[:,2]
                sorted_indices = np.argsort(azim_array)
                azim_array_sorted = azim_array[sorted_indices]
                polar_array_sorted = polar_array[sorted_indices]
                fil_phases_sorted = np.array([])

                #########################
                # Combine x and y into a single array
                pos_data = np.column_stack((polar_array_sorted, azim_array_sorted))

                # Specify the number of clusters you want
                n_clusters = int(self.nfil/10) 
                variance_array = np.zeros(n_clusters) # correlation within each cluster
                variance_array_angle = np.zeros(n_clusters)

                # Create and fit a K-Means model
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(pos_data)
                cluster_assignments = kmeans.labels_
                ##################

                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    fil_states_str = fil_states_f.readline()

                    if(i>=self.plot_start_frame):
                        fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                        fil_phases = fil_states[:self.nfil]
                        fil_angles = fil_states[self.nfil:]

                        '''
                        fil_angles_sorted = fil_angles[sorted_indices]
                        fil_phases_sorted = fil_phases[sorted_indices]
                        sin_phases_sorted = np.sin(fil_phases_sorted)
                        
                        # Coordination number 1
                        phase_diff = np.diff(sin_phases_sorted, prepend=sin_phases_sorted[-1])
                        corr = np.abs(phase_diff[:-1]) + np.abs(phase_diff[1:])
                        corr_array[i-self.plot_start_frame] = np.mean(corr)

                        # Coordination number 2
                        for m in range(n_clusters):
                            phases_in_group = sin_phases_sorted[np.where(cluster_assignments==m)]
                            variance_array[m] = np.var(phases_in_group)
                        corr_array2[i-self.plot_start_frame] = np.mean(variance_array)
                        '''

                        # Coordination number for angle
                        # for m in range(n_clusters):
                        #     angles_in_group = fil_angles_sorted[np.where(cluster_assignments==m)]
                        #     variance_array_angle[m] = np.var(angles_in_group)
                        
                        # corr_array_angle[i-self.plot_start_frame] = np.mean(variance_array_angle)
                        r_array[i-self.plot_start_frame] = np.abs(np.sum(np.exp(1j*fil_phases))/self.nfil)

                        # wavenumber_array[i-self.plot_start_frame] = fil_phases_sorted[0] - fil_phases_sorted[-1]

                color = cmap(self.index/self.num_sim)
                ax.plot(time_array, corr_array, color=color, label=f"index={self.index}")
                ax2.plot(time_array, corr_array2, color=color, label=f"index={self.index}")
                ax3.plot(time_array, r_array, color=color, label=f"index={self.index}")
                
                
            except:
                print("WARNING: " + self.simName + " not found.")
        
        ax.set_xlabel('t/T')
        ax.set_ylabel('Coordination number')
        ax.set_xlim(time_array[0], time_array[-1])
        ax.set_ylim(0)

        ax2.set_xlabel('t/T')
        ax2.set_ylabel('Coordination number 2')
        ax2.set_xlim(time_array[0], time_array[-1])
        ax2.set_ylim(0)

        
        ax3.set_ylim(0)
        ax3.set_xlabel('t/T')
        ax3.set_ylabel('<r>')
        ax3.set_xlim(time_array[0], time_array[-1])


        fig.legend()
        fig2.legend()
        fig3.legend()
        plt.tight_layout()
        # fig.savefig(f'fig/multi_coordination_parameter_one.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/multi_coordination_parameter_two.pdf', bbox_inches = 'tight', format='pdf')
        fig3.savefig(f'fig/multi_order_parameter_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        # plt.show()

    def multi_timing(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                with open(self.simName + '_time.dat', 'r') as file:
                    plot_time_frame = min(300, len(file.readlines()))
                timings_f = open(self.simName + '_time.dat', "r")

                time_array = np.arange(0, plot_time_frame)
                timings_array = np.zeros((len(time_array), 7))
                total_timing_array = np.zeros(len(time_array))

                for i in range(plot_time_frame):
                    print(" frame ", i, "/", plot_time_frame, "          ", end="\r")
                    timings_str = timings_f.readline()

                    timings = np.array(timings_str.split(), dtype=float)

                    timings_array[i] = timings[:7]
                for i in range(len(timings_array[0])):
                    # ax.plot(time_array, timings_array[:,i], label=labels[i])
                    total_timing_array += timings_array[:,i]
                ax.plot(time_array, total_timing_array, label=f"nfil={self.nfil} AR={self.ar}")
            except:
                print("WARNING: " + self.simName + " not found.")

        plt.tight_layout()
        plt.savefig(f'fig/multi_timing.png', bbox_inches = 'tight', format='png')
        plt.savefig(f'fig/multi_timing.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def multi_ciliate_svd(self):
         # Plotting
        # nrow = len(np.unique(self.pars_list['nfil']))
        # ncol = len(np.unique(self.pars_list['ar']))
        # spring_factor = self.pars_list['spring_factor'][0]

        ncol = self.ncol
        nrow = self.num_sim//ncol

        # nrow = int(self.num_sim**.5)
        # ncol = nrow + (1 if nrow**2 < self.num_sim else 0)
        fig, axs = plt.subplots(nrow, ncol, figsize=(18, 18), sharex=True, sharey=True)
    
        axs_flat = axs.ravel()

        for ind, ax in enumerate(axs_flat):
            if (ind < self.num_sim):
                try:
                    self.index = ind
                    self.select_sim()

                    fil_phases_f = open(self.simName + '_filament_phases.dat', "r")
                    fil_angles_f = open(self.simName + '_filament_shape_rotation_angles.dat', "r")


                    nfil = self.nfil
                    n_snapshots = min(4800, self.plot_end_frame)
                    start = self.plot_end_frame - n_snapshots
                    X = np.zeros((nfil, n_snapshots))
                    X_angle = np.zeros((nfil, n_snapshots))

                    fil_references_sphpolar = np.zeros((nfil,3))
                    for fil in range(nfil):
                        fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
                    azim_array = fil_references_sphpolar[:,1]
                    polar_array = fil_references_sphpolar[:,2]
                    sorted_indices = np.argsort(azim_array)
                    azim_array_sorted = azim_array[sorted_indices]
                    polar_array_sorted = polar_array[sorted_indices]

                    for i in range(self.plot_end_frame):
                        print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                        fil_phases_str = fil_phases_f.readline()
                        fil_angles_str = fil_angles_f.readline()
                        
                        if(i>=start):
                            fil_phases = np.array(fil_phases_str.split()[1:], dtype=float)
                            fil_phases_sorted = fil_phases[sorted_indices]
                            fil_phases_sorted = util.box(fil_phases_sorted, 2*np.pi)

                            fil_phases_sorted = np.sin(fil_phases_sorted)

                            fil_angles = np.array(fil_angles_str.split()[1:], dtype=float)
                            fil_angles_sorted = fil_angles[sorted_indices]

                            X[:,i-start] = fil_phases_sorted[:nfil]
                            X_angle[:,i-start] = fil_angles_sorted[:nfil]
                    
                    print("save")
                    np.savetxt(f'phase_data_20231201/spring_constant{self.spring_factor}/X_phase_index{self.index}.txt', X, delimiter=', ')
                    # np.savetxt(f'phase_data_20231107/by_index/spring_constant{spring_factor}/X_rotation_angle_index{self.index}.txt', X_angle, delimiter=', ')
                    np.savetxt(f'phase_data_20231201/spring_constant{self.spring_factor}/azim_pos_index{self.index}.txt', azim_array_sorted, delimiter=', ')
                    np.savetxt(f'phase_data_20231201/spring_constant{self.spring_factor}/polar_pos_index{self.index}.txt', polar_array_sorted, delimiter=', ')

                    # np.savetxt(f'phase_data/by_pars/spring_constant{spring_factor}/X_phase_nfil{nfil}_rol{self.ar}_spring{self.spring_factor}.txt', X, delimiter=', ')
                    # np.savetxt(f'phase_data/by_pars/spring_constant{spring_factor}/X_rotation_angle_nfil{nfil}_rol{self.ar}_spring{self.spring_factor}.txt', X_angle, delimiter=', ')
                    # np.savetxt(f'phase_data/by_pars/spring_constant{spring_factor}/azim_pos_nfil{nfil}_rol{self.ar}_spring{self.spring_factor}.txt', azim_array_sorted, delimiter=', ')
                    # np.savetxt(f'phase_data/by_pars/spring_constant{spring_factor}/polar_pos_nfil{nfil}_rol{self.ar}_spring{self.spring_factor}.txt', polar_array_sorted, delimiter=', ')


                    U, sigma, V = np.linalg.svd(X, full_matrices=False)
                    Sigma = np.diag(sigma)

                    pc = U @ Sigma
                    pa = V

                    # for fil in range(num_fil):
                    #     abs_pc = np.abs(pc[fil][:nfil])
                    #     ax.plot(np.cumsum(abs_pc)/np.sum(abs_pc), label=f'fil {fil}')
                    # ax.set_xlabel('Mode')
                    # ax.set_ylabel('Accumulated |weight| fraction')
                    # ax.legend()

                except:
                    print("WARNING: " + self.simName + " not found.")

        # plt.tight_layout()
        # plt.savefig(f'fig/multi_svd.png', bbox_inches = 'tight', format='png')
        # plt.savefig(f'fig/multi_svd.pdf', bbox_inches = 'tight', format='pdf')
        # plt.show()
           
    def multi_copy_phases(self):
        for ind in range(self.num_sim):
            self.index = ind
            try:
                self.select_sim()
                afix = int(self.index)

                input_filenames = [self.simName + '_true_states.dat',
                                self.simName + '_body_states.dat']
                
                output_filenames = [self.dir + f"psi{afix}.dat",
                                    self.dir + f"bodystate{afix}.dat",
                                    ]
                output_filenames = [f"data/tilt_test/output/{self.date}/" + f"psi{afix}.dat",
                                    f"data/tilt_test/output/{self.date}/" + f"bodystate{afix}.dat",
                                    ]
        
                for i, name in enumerate(input_filenames):
                    input_filename = name
                    output_filename = output_filenames[i]
                    try:
                        # Open the input file in read mode
                        with open(input_filename, 'r') as input_file:
                            # Read all lines from the file
                            lines = input_file.readlines()

                            # Check if the file is not empty
                            if lines:
                                # Extract the last line
                                last_line = lines[-1]

                                data = np.array(last_line.split()[1:], dtype=float)
                                # data[:self.nfil] = util.box(data[:self.nfil], 2*np.pi)

                                if input_filename == self.simName + '_true_states.dat':
                                    data = np.concatenate(([self.spring_factor], data))

                                np.savetxt(output_filename, data, delimiter=' ', newline=' ')

                                print(f"Success: last line copied from '{input_filename}' to '{output_filename}'.")
                            else:
                                print(f"The file '{input_filename}' is empty.")
                    except FileNotFoundError:
                        print(f"Error: The file '{input_filename}' does not exist.")
            except:
                print("WARNING: " + self.simName + " not found.")

# Summary plot
    def summary_ciliate_speed(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)

        nfil_list = np.array(self.pars_list['nfil'])
        ar_list = np.array(self.pars_list['ar'])
        sphere_r_list = np.zeros(self.num_sim)
        speed_list = np.zeros(self.num_sim)
        angular_speed_list = np.zeros(self.num_sim)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                body_vels_f = open(self.simName + '_body_vels.dat', "r")

                body_speed_array = np.zeros(self.frames)
                body_angular_speed_array = np.zeros(self.frames)

                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    body_vels_str = body_vels_f.readline()

                    if(i>=self.plot_start_frame):
                        body_vels = np.array(body_vels_str.split(), dtype=float)

                        body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                        body_angular_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[3:6]*body_vels[3:6], 0))

                speed_list[ind] = np.mean(body_speed_array)
                angular_speed_list[ind] = np.mean(body_angular_speed_array)
                sphere_r_list[ind] = self.radius
            except:
                print("WARNING: " + self.simName + " not found.")
        
        colormap = 'Greys'
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        vmin, vmax = np.min(speed_list), np.max(speed_list)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig2.colorbar(sm)
        cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
        cbar.set_label(r"Speed")

        ax2.scatter(self.pars_list['nfil'], self.pars_list['ar'], s=100, c=speed_list, edgecolors='red', linewidths=0.1, cmap=colormap)
        ax2.set_xlabel("Nfil")
        ax2.set_ylabel("R/L")
        fig2.savefig(f'fig/multi_ciliate_speed_summary_heatmap.png', bbox_inches = 'tight', format='png')
        fig2.savefig(f'fig/multi_ciliate_speed_summary_heatmap.pdf', bbox_inches = 'tight', format='pdf')

        vmin, vmax = np.min(angular_speed_list), np.max(angular_speed_list)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig3.colorbar(sm)
        cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
        cbar.set_label(r"Angular speed")

        ax3.scatter(self.pars_list['nfil'], self.pars_list['ar'], s=100, c=angular_speed_list, edgecolors='red', linewidths=0.1, cmap=colormap)
        ax3.set_xlabel("Nfil")
        ax3.set_ylabel("R/L")
        fig3.savefig(f'fig/multi_ciliate_angular_speed_summary_heatmap.png', bbox_inches = 'tight', format='png')
        fig3.savefig(f'fig/multi_ciliate_angular_speed_summary_heatmap.pdf', bbox_inches = 'tight', format='pdf')
        

        ax.scatter(nfil_list, speed_list, label=f"nfil={self.nfil} AR={self.ar}")
        ax.set_ylabel("Velocity")
        ax.set_xlabel("Number of filaments")
        fig.tight_layout()
        fig.savefig(f'fig/multi_ciliate_speed_summary.png', bbox_inches = 'tight', format='png')
        fig.savefig(f'fig/multi_ciliate_speed_summary.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def summary_timing(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        num_particle_list = np.array(self.pars_list['nfil'])*20 + np.array(self.pars_list['nblob'])
        total_timing_list = np.zeros((self.num_sim, 7))

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                with open(self.simName + '_time.dat', 'r') as file:
                    plot_time_frame = min(300, len(file.readlines()))
                timings_f = open(self.simName + '_time.dat', "r")

                for i in range(plot_time_frame):
                    print(" frame ", i, "/", plot_time_frame, "          ", end="\r")
                    timings_str = timings_f.readline()

                    timings = np.array(timings_str.split()[:-1], dtype=float)

                    total_timing_list[ind] += timings
            except:
                print("WARNING: " + self.simName + " not found.")


        ax.scatter(num_particle_list, total_timing_list[:,3]/plot_time_frame, label="HI solver")
        ax.scatter(num_particle_list, np.sum(total_timing_list, axis=1)/plot_time_frame, label="Total")
        # ax.set_ylim(0)
        # ax.set_yscale('log')
        ax.set_ylabel("Compute time per time step/s")
        ax.set_xlabel("Number of particles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fig/multi_timing_summary.png', bbox_inches = 'tight', format='png')
        plt.savefig(f'fig/multi_timing_summary.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def summary_check_overlap(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for ind in range(self.num_sim):
            # try:
            self.index = ind
            self.select_sim()

            seg_states_f = open(self.simName + '_seg_states.dat', "r")
            body_states_f = open(self.simName + '_body_states.dat', "r")

            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                segs_list = np.zeros((int(self.nfil*self.nseg), 3))
                blobs_list = np.zeros((int(self.nblob), 3))

                body_states_str = body_states_f.readline()
                seg_states_str = seg_states_f.readline()
                
                if(i==self.plot_end_frame-1):
                    body_states = np.array(body_states_str.split()[1:], dtype=float)
                    seg_states = np.array(seg_states_str.split()[1:], dtype=float)

                    for swim in range(int(self.pars['NSWIM'])):
                        body_pos = body_states[7*swim : 7*swim+3]
                        R = util.rot_mat(body_states[7*swim+3 : 7*swim+7])
                        R = np.linalg.inv(R)
                        R = np.eye(3)
                        for blob in range(int(self.pars['NBLOB'])):
                            blob_x, blob_y, blob_z = util.blob_point_from_data(body_states[7*swim : 7*swim+7], self.blob_references[3*blob:3*blob+3])
                            blobs_list[blob] = blob_x, blob_y, blob_z
                        for fil in range(int(self.pars['NFIL'])):
                            fil_i = int(3*fil*self.pars['NSEG'])
                            old_seg_pos = seg_states[fil_i : fil_i+3]
                            segs_list[fil*self.nseg] = old_seg_pos
                            for seg in range(1, int(self.pars['NSEG'])):
                                seg_pos = seg_states[fil_i+3*(seg) : fil_i+3*(seg+1)]
                                segs_list[fil*self.nseg + seg] = seg_pos
                    threshold = 0.95
                    colliding_indices, colliding_particles = util.label_colliding_particles_with_3d_cell_list(segs_list, 5, threshold*float(self.pars['RSEG']))
                    if not colliding_indices:
                        print(f'index={ind} - No overlapping at threshold {threshold}\n')
                    else:
                        print(f'index={ind} - Overlapping at threshold {threshold}\n')
            # except:
            #     print("WARNING: " + self.simName + " not found.")
        
        # ax.scatter(nfil_list, speed_list, label=f"nfil={self.nfil} AR={self.ar}")
        # ax.set_ylabel("Velocity")
        # ax.set_xlabel("Number of filaments")
        # fig.tight_layout()
        # fig.savefig(f'fig/multi_ciliate_speed_summary.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def summary_ciliate_dissipation(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(1,1,1)

        nfil_list = np.array(self.pars_list['nfil'])
        ar_list = np.array(self.pars_list['ar'])
        sphere_r_list = np.zeros(self.num_sim)
        speed_list = np.zeros(self.num_sim)
        angular_speed_list = np.zeros(self.num_sim)
        dissipation_list = np.zeros(self.num_sim)
        efficiency_list = np.zeros(self.num_sim)
        k_list = np.zeros(self.num_sim)

        # nrow = int(self.num_sim**.5)
        # ncol = nrow + (1 if nrow**2 < self.num_sim else 0)
        ncol = self.ncol
        nrow = self.num_sim//ncol

        print(f'nrow = {nrow} ncol = {ncol}')
        # spring_factor = self.pars_list['spring_factor'][0]

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")
                
                body_speed_array = np.zeros(self.frames)
                body_angular_speed_array = np.zeros(self.frames)
                dissipation_array = np.zeros(self.frames)

                blob_references_str = blob_references_f.readline()
                blob_references = np.array(blob_references_str.split(), dtype=float)
                blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))
            
                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    seg_forces_str = seg_forces_f.readline()
                    seg_vels_str = seg_vels_f.readline()
                    blob_forces_str = blob_forces_f.readline()
                    body_vels_str = body_vels_f.readline()

                    if(i>=self.plot_start_frame):
                        seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                        seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                        blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                        body_vels= np.array(body_vels_str.split(), dtype=float)

                        seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                        body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                        blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                        body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                        body_angular_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[3:6]*body_vels[3:6], 0))
                        dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)

                speed_list[ind] = np.mean(body_speed_array)
                angular_speed_list[ind] = np.mean(body_angular_speed_array)
                dissipation_list[ind] = np.mean(dissipation_array)
                sphere_r_list[ind] = self.radius
                k_list[ind] = self.spring_factor
            except:
                print("WARNING: " + self.simName + " not found.")

        efficiency_list = 6*np.pi*sphere_r_list*speed_list**2/dissipation_list

        colormap = 'Greys'
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable


        ax.scatter(k_list, dissipation_list, label=f"k={self.spring_factor} AR={self.ar}")
        ax.set_ylabel("Dissipation")
        # ax.set_xlabel(r"$N_f$")
        ax.set_xlabel(r"$k$")
        fig.tight_layout()
        fig.savefig(f'fig/ciliate_dissipation_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

        # #Speed
        # vmin, vmax = np.min(speed_list), np.max(speed_list)
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # sm = ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])
        # cbar = fig2.colorbar(sm)
        # cbar.ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        # cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
        # cbar.set_label(r"Speed")

        # ax2.scatter(self.pars_list['spring_factor'], self.pars_list['ar'], s=100, c=speed_list, edgecolors='red', linewidths=0.1, cmap=colormap)
        # # ax2.set_xlabel(r"$N_f$")
        # ax2.set_xlabel(r"$k$")
        # ax2.set_ylabel("R/L")
        # fig2.savefig(f'fig/multi_ciliate_speed_summary_heatmap_{self.date}.pdf', bbox_inches = 'tight', format='pdf')

        # #Angular speed
        # vmin, vmax = np.min(angular_speed_list), np.max(angular_speed_list)
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # sm = ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])
        # cbar = fig3.colorbar(sm)
        # cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
        # cbar.set_label(r"Angular speed")

        # ax3.scatter(self.pars_list['nfil'], self.pars_list['ar'], s=100, c=angular_speed_list, edgecolors='red', linewidths=0.1, cmap=colormap)
        # # ax3.set_xlabel(r"$N_f$")
        # ax3.set_xlabel(r"$k$")
        # ax3.set_ylabel("R/L")
        # fig3.savefig(f'fig/multi_ciliate_angular_speed_summary_heatmap_{self.date}.pdf', bbox_inches = 'tight', format='pdf')

        # #Dissipation
        # vmin, vmax = np.min(dissipation_list), np.max(dissipation_list)
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # sm = ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])
        # cbar = fig4.colorbar(sm)
        # cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
        # cbar.set_label(r"Dissipation")

        # ax4.scatter(self.pars_list['nfil'], self.pars_list['ar'], s=100, c=dissipation_list, edgecolors='red', linewidths=0.1, cmap=colormap)
        # ax4.set_xlabel("Nfil")
        # ax4.set_ylabel("R/L")
        # fig4.savefig(f'fig/multi_ciliate_dissipation_summary_heatmap_{self.date}.pdf', bbox_inches = 'tight', format='pdf')

        # #Efficiency
        # vmin, vmax = np.min(efficiency_list), np.max(efficiency_list)
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # sm = ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])
        # cbar = fig5.colorbar(sm)
        # cbar.ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        # cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
        # cbar.set_label(r"Efficiency")

        # ax5.scatter(self.pars_list['nfil'], self.pars_list['ar'], s=100, c=efficiency_list, edgecolors='red', linewidths=0.1, cmap=colormap)
        # ax5.set_xlabel("Nfil")
        # ax5.set_ylabel("R/L")
        # fig5.savefig(f'fig/multi_ciliate_efficiency_summary_heatmap_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        
    def summary_ciliate_density(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)
        
        ncol = self.ncol
        nrow = self.num_sim//ncol
        print(f'nrow = {nrow} ncol = {ncol}')

        nfil_list = np.array(self.pars_list['nfil'])
        density_list = np.zeros(self.num_sim)
        ar_list = np.zeros(self.num_sim)
        fillen_list = np.zeros(self.num_sim)
        sphere_r_list = np.zeros(self.num_sim)
        speed_list = np.zeros(self.num_sim)
        angular_speed_list = np.zeros(self.num_sim)
        dissipation_list = np.zeros(self.num_sim)
        efficiency_list = np.zeros(self.num_sim)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")

            except:
                print("WARNING: " + self.simName + " not found.")
                continue
                # raise FileExistsError("WARNING: " + self.simName + " not found.")
                
            body_speed_array = np.zeros(self.frames)
            dissipation_array = np.zeros(self.frames)
            efficiency_array = np.zeros(self.frames)

            blob_references_str = blob_references_f.readline()
            blob_references = np.array(blob_references_str.split(), dtype=float)
            blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))
        
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                seg_forces_str = seg_forces_f.readline()
                seg_vels_str = seg_vels_f.readline()
                blob_forces_str = blob_forces_f.readline()
                body_vels_str = body_vels_f.readline()

                if(i>=self.plot_start_frame):
                    seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                    seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                    blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                    body_vels= np.array(body_vels_str.split(), dtype=float)

                    seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                    seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                    blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                    body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                    blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                    body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                    dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)
                    # efficiency_array[i-self.plot_start_frame] = 6*np.pi*self.radius*body_speed_array[i-self.plot_start_frame]**2/dissipation_array[i-self.plot_start_frame]

            seg_forces_f.close()
            seg_vels_f.close()
            blob_forces_f.close()
            blob_references_f.close()
            body_vels_f.close()
            
            speed_list[ind] = np.mean(body_speed_array)
            dissipation_list[ind] = np.mean(dissipation_array)
            # efficiency_array[ind] = np.mean(efficiency_array)
            sphere_r_list[ind] = self.radius
            ar_list[ind] = self.ar
            fillen_list[ind] = self.fillength
            density_list[ind] = self.fildensity

        efficiency_list = 6*np.pi*sphere_r_list*speed_list**2/dissipation_list
        
        linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', '' ]
        
        for i in range(ncol):
            ax.plot(nfil_list[i::ncol], speed_list[i::ncol]/fillen_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2e}")
            ax2.plot(nfil_list[i::ncol], dissipation_list[i::ncol]/fillen_list[i::ncol]**3, marker='+', linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2e}")
            ax3.plot(nfil_list[i::ncol], efficiency_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2e}")
            ax4.plot(nfil_list[i::ncol], dissipation_list[i::ncol]/nfil_list[i::ncol]/fillen_list[i::ncol]**3, marker='+', linestyle=linestyle_list[i], c='black', label=f"density={density_list[i]:.2e}")
            

        fig.legend()
        fig2.legend()
        fig3.legend()
        fig4.legend()
        ax.set_xlabel(r'$N_{fil}$')
        ax.set_ylabel(r'$<VT/L>$')
        ax2.set_xlabel(r'$N_{fil}$')
        ax2.set_ylabel(r'$<PT^2/\mu L^3>$')
        ax3.set_xlabel(r'$N_{fil}$')
        ax3.set_ylabel(r'$<Efficiency>$')
        ax4.set_xlabel(r'$N_{fil}$')
        ax4.set_ylabel(r'$<PT^2/\mu L^3 N_{fil}>$')

        # fig.savefig(f'fig/ciliate_speed_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/ciliate_dissipation_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        # fig3.savefig(f'fig/ciliate_efficiency_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        # fig4.savefig(f'fig/ciliate_dissipation_per_cilium_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')

        plt.show()

    def summary_ciliate_k(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)

        ncol = 1
        nrow = self.num_sim//ncol
        print(f'nrow = {nrow} ncol = {ncol}')

        k_list = np.array(self.pars_list['spring_factor'])
        nfil_list = np.array(self.pars_list['nfil'])
        density_list = np.zeros(self.num_sim)
        ar_list = np.zeros(self.num_sim)
        fillen_list = np.zeros(self.num_sim)
        sphere_r_list = np.zeros(self.num_sim)
        speed_list = np.zeros(self.num_sim)
        angular_speed_list = np.zeros(self.num_sim)
        dissipation_list = np.zeros(self.num_sim)
        efficiency_list = np.zeros(self.num_sim)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")

            except:
                print("WARNING: " + self.simName + " not found.")
                continue
                # raise FileExistsError("WARNING: " + self.simName + " not found.")
                
            body_speed_array = np.zeros(self.frames)
            dissipation_array = np.zeros(self.frames)
            efficiency_array = np.zeros(self.frames)

            blob_references_str = blob_references_f.readline()
            blob_references = np.array(blob_references_str.split(), dtype=float)
            blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))
        
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                seg_forces_str = seg_forces_f.readline()
                seg_vels_str = seg_vels_f.readline()
                blob_forces_str = blob_forces_f.readline()
                body_vels_str = body_vels_f.readline()

                if(i>=self.plot_start_frame):
                    seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                    seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                    blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                    body_vels= np.array(body_vels_str.split(), dtype=float)

                    seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                    seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                    blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                    body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                    blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                    body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                    dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)
                    # efficiency_array[i-self.plot_start_frame] = 6*np.pi*self.radius*body_speed_array[i-self.plot_start_frame]**2/dissipation_array[i-self.plot_start_frame]

            seg_forces_f.close()
            seg_vels_f.close()
            blob_forces_f.close()
            blob_references_f.close()
            body_vels_f.close()
            
            speed_list[ind] = np.mean(body_speed_array)
            dissipation_list[ind] = np.mean(dissipation_array)
            # efficiency_array[ind] = np.mean(efficiency_array)
            sphere_r_list[ind] = self.radius
            ar_list[ind] = self.ar
            fillen_list[ind] = self.fillength
            density_list[ind] = self.fildensity

        efficiency_list = 6*np.pi*sphere_r_list*speed_list**2/dissipation_list
        
        linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', '' ]
        
        for i in range(ncol):
            label = f'k='
            ax.plot(k_list[i::ncol], speed_list[i::ncol]/fillen_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black')
            ax2.plot(k_list[i::ncol], dissipation_list[i::ncol]/fillen_list[i::ncol]**3, marker='+', linestyle=linestyle_list[i], c='black')
            ax3.plot(k_list[i::ncol], efficiency_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black')
            ax4.plot(k_list[i::ncol], dissipation_list[i::ncol]/nfil_list[i::ncol]/fillen_list[i::ncol]**3, marker='+', linestyle=linestyle_list[i], c='black')
            
            np.savetxt(f'dis_data/speed_{self.date}.txt', (speed_list[i::ncol]/fillen_list[i::ncol])[:24], delimiter=', ')
            np.savetxt(f'dis_data/dissipation_{self.date}.txt', (dissipation_list[i::ncol]/fillen_list[i::ncol]**3)[:24], delimiter=', ')
            np.savetxt(f'dis_data/efficiency_{self.date}.txt', (efficiency_list[i::ncol])[:24], delimiter=', ')
            np.savetxt(f'dis_data/k_{self.date}.txt', (k_list[i::ncol])[:24], delimiter=', ')
        

        # fig.legend()
        # fig2.legend()
        # fig3.legend()
        # fig4.legend()
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$<VT/L>$')
        ax2.set_xlabel(r'$k$')
        ax2.set_ylabel(r'$<PT^2/\mu L^3>$')
        ax3.set_xlabel(r'$k$')
        ax3.set_ylabel(r'$<Efficiency>$')
        ax4.set_xlabel(r'$k$')
        ax4.set_ylabel(r'$<PT^2/\mu L^3 N_{fil}>$')

        fig.savefig(f'fig/ciliate_speed_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig(f'fig/ciliate_dissipation_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig3.savefig(f'fig/ciliate_efficiency_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig4.savefig(f'fig/ciliate_dissipation_per_cilium_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')

        plt.show()

    def summary_ciliate_resolution(self):
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)

        ncol = 1
        nrow = self.num_sim//ncol
        print(f'nrow = {nrow} ncol = {ncol}')

        k_list = np.array(self.pars_list['spring_factor'])
        nfil_list = np.array(self.pars_list['nfil'])
        nblob_list = np.array(self.pars_list['nblob'])
        density_list = np.zeros(self.num_sim)
        ar_list = np.zeros(self.num_sim)
        fillen_list = np.zeros(self.num_sim)
        sphere_r_list = np.zeros(self.num_sim)
        speed_list = np.zeros(self.num_sim)
        angular_speed_list = np.zeros(self.num_sim)
        dissipation_list = np.zeros(self.num_sim)
        efficiency_list = np.zeros(self.num_sim)
        error_list = np.zeros(self.num_sim)

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")

            except:
                print("WARNING: " + self.simName + " not found.")
                continue
                # raise FileExistsError("WARNING: " + self.simName + " not found.")
                
            body_speed_array = np.zeros(self.frames)
            dissipation_array = np.zeros(self.frames)
            efficiency_array = np.zeros(self.frames)

            blob_references_str = blob_references_f.readline()
            blob_references = np.array(blob_references_str.split(), dtype=float)
            blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))
        
            for i in range(self.plot_end_frame):
                print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                seg_forces_str = seg_forces_f.readline()
                seg_vels_str = seg_vels_f.readline()
                blob_forces_str = blob_forces_f.readline()
                body_vels_str = body_vels_f.readline()

                if(i>=self.plot_start_frame):
                    phases = 0
                    seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                    seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                    blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                    body_vels= np.array(body_vels_str.split(), dtype=float)

                    seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                    seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                    blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                    body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                    blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                    body_speed_array[i-self.plot_start_frame] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                    dissipation_array[i-self.plot_start_frame] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)
                    # efficiency_array[i-self.plot_start_frame] = 6*np.pi*self.radius*body_speed_array[i-self.plot_start_frame]**2/dissipation_array[i-self.plot_start_frame]

            seg_forces_f.close()
            seg_vels_f.close()
            blob_forces_f.close()
            blob_references_f.close()
            body_vels_f.close()
            
            speed_list[ind] = np.mean(body_speed_array)
            dissipation_list[ind] = np.mean(dissipation_array)
            # efficiency_array[ind] = np.mean(efficiency_array)
            sphere_r_list[ind] = self.radius
            ar_list[ind] = self.ar
            fillen_list[ind] = self.fillength
            density_list[ind] = self.fildensity

        efficiency_list = 6*np.pi*sphere_r_list*speed_list**2/dissipation_list

        ref_error = dissipation_list[14]
        error_list = np.abs(dissipation_list - ref_error)/ref_error
        
        linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', '' ]
        
        for i in range(ncol):
            label = f'k='
            ax.plot(nblob_list[i::ncol], speed_list[i::ncol]/fillen_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black')
            ax2.plot(nblob_list[i::ncol], dissipation_list[i::ncol]/fillen_list[i::ncol]**3, marker='+', linestyle=linestyle_list[i], c='black')
            ax3.plot(nblob_list[i::ncol], efficiency_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black')
            
            ax4.plot(nblob_list[i::ncol], error_list[i::ncol], marker='+', linestyle=linestyle_list[i], c='black')
            
            np.save(f"{self.dir}nblob_data_{i}.npy", nblob_list[i::ncol])
            np.save(f"{self.dir}speed_data_{i}.npy", speed_list[i::ncol]/fillen_list[i::ncol])
            np.save(f"{self.dir}dissipation_data_{i}.npy", dissipation_list[i::ncol]/fillen_list[i::ncol]**3)
            np.save(f"{self.dir}efficiency_data_{i}.npy", efficiency_list[i::ncol])

            # np.savetxt(f'dis_data/speed_{self.date}.txt', (speed_list[i::ncol]/fillen_list[i::ncol])[:24], delimiter=', ')
            # np.savetxt(f'dis_data/dissipation_{self.date}.txt', (dissipation_list[i::ncol]/fillen_list[i::ncol]**3)[:24], delimiter=', ')
            # np.savetxt(f'dis_data/efficiency_{self.date}.txt', (efficiency_list[i::ncol])[:24], delimiter=', ')
            # np.savetxt(f'dis_data/k_{self.date}.txt', (k_list[i::ncol])[:24], delimiter=', ')
        

        # fig.legend()
        # fig2.legend()
        # fig3.legend()
        # fig4.legend()
        ax.set_xscale("log")
        ax2.set_xscale("log")
        ax3.set_xscale("log")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
            
        ax.set_xlabel(r'$N_{blob}$')
        ax.set_ylabel(r'$<VT/L>$')
        ax2.set_xlabel(r'$N_{blob}$')
        ax2.set_ylabel(r'$<PT^2/\mu L^3>$')
        ax3.set_xlabel(r'$N_{blob}$')
        ax3.set_ylabel(r'$<Efficiency>$')
        ax4.set_xlabel(r'$N_{blob}$')
        ax4.set_ylabel(r'$\% error in dissipation$')

        fig.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()

        fig.savefig(f'fig/ciliate_speed_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig(f'fig/ciliate_dissipation_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig3.savefig(f'fig/ciliate_efficiency_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
        fig4.savefig(f'fig/ciliate_error_summary_{self.date}.pdf', bbox_inches = 'tight', format='pdf')

        plt.show()


# Special plot
    def ishikawa(self):
        top_dir = "data/expr_sims/ishikawa_expr/"
        vel_dirs = dirs = [ "k0.0/",
                            "k0.5/",
                            "k1.0/",
                            "k1.5/",
                            "k2.0/"]
        
        # dissipation_dirs = ["k0.0/",
        #                     "k1.0/"]
        
        N_dirs = ["k0.0N162/",
                  "k0.0N636/",
                  "k0.0N2520/",]
        # N_dirs = ["k0.0N162_RoL20/",
        #           "k0.0N636_RoL20/",
        #           "k0.0N2520_RoL20/",]

        dirs = ["k0.0/",
                "k0.5/",
                "k1.0/",
                "k1.5/",
                "k2.0/",
                "k0.0N162/",
                "k0.0N636/",
                "k0.0N2520/"]
        
        top_dir = "data/expr_sims/20231212_ishikawa/"
        N_dirs = ["N160_hres/",
                  "N640_hres/",
                  "N2560_hres/",]
        
        ls = ['solid', 'dashed', 'dotted']
        markers = ["^", "s", "d"]
        labels = [r"$k=0$",r"$k=0.5$",r"$k=1$",r"$k=1.5$",r"$k=2$",]
        colors = ["black","red","green","blue","purple"]
        dissipation_labels = [r"$N=162$",r"$N=636$",r"$N=2520$"]
        dissipation_colors = ["green","black","red"]
        vel_marker = 0
        dissipation_marker = 0

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.4, 4.5)
        ax.set_ylabel(r"$V_zT/L$")
        ax.set_xlabel(r"$t/T$")

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 80000)
        ax2.set_ylabel(r"$PT^2/\mu L^3$")
        ax2.set_xlabel(r"$t/T$")

        for ni, directory in enumerate(N_dirs):
            try:
                self.dir = top_dir + directory
                self.read_rules()
                self.select_sim()
                L = self.pars['FIL_LENGTH']
                print(f'dt={self.dt}')
                print(f'L={L}')
                start_time = 0
                end_time = min(301, sum(1 for line in open(self.simName + '_body_states.dat')))

                body_states_f = open(self.simName + '_body_states.dat', "r")
                body_vels_f = open(self.simName + '_body_vels.dat', "r")
                seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                blob_references_f = open(self.simName + '_blob_references.dat', "r")

                time_array = np.arange(start_time, end_time)
                body_pos_array = np.zeros((len(time_array), 3))
                body_vel_array = np.zeros((len(time_array), 6))
                body_speed_array = np.zeros(len(time_array))
                body_angular_speed_array = np.zeros(len(time_array))
                dissipation_array = np.zeros(len(time_array))

                blob_references_str = blob_references_f.readline()
                blob_references = np.array(blob_references_str.split(), dtype=float)
                blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))

                for i in range(end_time):
                    print(" frame ", i, "/", end_time, "          ", end="\r")
                    seg_forces_str = seg_forces_f.readline()
                    seg_vels_str = seg_vels_f.readline()
                    blob_forces_str = blob_forces_f.readline()
                    body_vels_str = body_vels_f.readline()

                    if(i>=start_time):
                        seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                        seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                        blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                        body_vels = np.array(body_vels_str.split(), dtype=float)
                        

                        seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                        blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                        body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                        blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                        body_vel_array[i-start_time] = body_vels
                        body_speed_array[i-start_time] = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                        body_angular_speed_array[i-start_time] = np.sqrt(np.sum(body_vels[3:6]*body_vels[3:6], 0))
                        dissipation_array[i-start_time] = np.sum(blob_forces * blob_vels) + np.sum(seg_forces[:, 0:3] * seg_vels[:, 0:3])

                # if directory in vel_dirs:
                #     ax.plot(time_array/30., body_vel_array[:,2]/L, label=labels[vel_marker], c=colors[vel_marker])
                #     vel_marker += 1
                if directory in N_dirs:
                    ax.plot(time_array*self.dt, body_vel_array[:,2]/L, label=dissipation_labels[dissipation_marker], c=dissipation_colors[dissipation_marker])
                    ax2.plot(time_array*self.dt, dissipation_array/L**3, label=dissipation_labels[dissipation_marker], c=dissipation_colors[dissipation_marker])
                    dissipation_marker +=1
            except:
                print("WARNING: " + self.dir + "rules.ini not found.")
        
        # Plot the comparison data
        # JFM 2019 velocity
        directory = 'pyfile/analysis/ishikawa_data/'
        files = ['k0.0.csv', 'k0.5.csv', 'k1.0.csv', 'k1.5.csv', 'k2.0.csv']
        files = ['vel_k0.0N162.csv', 'vel_k0.0N636.csv', 'vel_k0.0N2520.csv']
        for i, filename in enumerate(files):
            try:
                file = open(directory + filename, mode='r')
                df = pd.read_csv(directory + filename, header=None)
                data = df.to_numpy()
                x, y = data[:,0], data[:,1]

                ax.plot(x, y, ls='dotted', c=dissipation_colors[i], alpha=0.5)
            except:
                print("WARNING: " + directory + filename + " not found.")
        # PNAS 2020 dissipation
        files = ['dissipation_k0.0N162.csv', 'dissipation_k0.0N636.csv', 'dissipation_k0.0N2520.csv']
        for i, filename in enumerate(files):
            try:
                file = open(directory + filename, mode='r')
                df = pd.read_csv(directory + filename, header=None)
                data = df.to_numpy()
                x, y = data[:,0], data[:,1]

                ax2.plot(x, y, ls='dotted', c=dissipation_colors[i], alpha=0.5)
            except:
                print("WARNING: " + directory + filename + " not found.")


        # Make legends
        legend_label = r'$Ito\ et al.\ (2019)$'
        legend_label = r'$Omori\ et al.\ (2020)$'
        legend1 = ax.legend()
        line1a, = ax.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$data$' )
        line1b, = ax.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=legend_label)
        ax.legend(handles = [line1a, line1b], loc='upper left')
        ax.add_artist(legend1)
        
        legend2 = ax2.legend()
        line2a, = ax2.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$data$' )
        line2b, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=legend_label)
        ax2.legend(handles = [line2a, line2b], loc='upper left')
        ax2.add_artist(legend2)

        fig.savefig(f'fig/PNAS_comparison_vel.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig(f'fig/PNAS_comparison_dissipation.pdf', bbox_inches = 'tight', format='pdf')
        # fig.savefig(f'fig/ishikawa_vel{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        # fig2.savefig(f'fig/ishikawa_dissipation{self.nfil}fil.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def view_solution(self):
        # Plotting
        colormap = 'cividis'
        colormap = 'twilight_shifted'

        sim_dir = '20240320_JFNK_d'
        dir = 'data/JFNK/' + sim_dir

        fil_references = myIo.read_fil_references(dir + '/fil_references.dat')
        input_filename = dir + '/psi_guess159_fullrange.dat'
        nfil = int(len(fil_references)/3)

        with open(input_filename, 'r') as input_file:
            lines = input_file.readlines()

        nsol = len(lines)
        nrow = int(nsol**.5)
        ncol = nrow + (1 if nrow**2 <nsol else 0)
        print(f'num sol = {nsol} nrow = {nrow} ncol = {ncol}')

        fil_references_sphpolar = np.zeros((nfil,3))
        for i in range(nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(fil_references[3*i: 3*i+3])

        T_array = np.zeros(nsol)
        k_array = np.zeros(nsol)
        diff_array = np.zeros(nsol)
        
        fig, axs = plt.subplots(nrow, ncol,sharex=True, sharey=True)
        fig2 = plt.figure()
        ax2  = fig2.add_subplot()

        axs_flat = axs.ravel()
        import scipy.interpolate

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        vmin = 0
        vmax = 2*np.pi
        if(self.angle):
            vmin = -.2*np.pi
            vmax = .2*np.pi

        for ind, ax in enumerate(axs_flat):
            ax.invert_yaxis()
            if (ind < nsol):
                if(ind > 0):
                    old_data = data
                data = np.array(lines[ind].split(), dtype=float)
                k = data[0]
                T = data[1]
                phase = data[2:2+nfil]
                phase = util.box(phase, 2*np.pi)
                angle = data[2+nfil:]


                k_array[ind] = k
                T_array[ind] = T
                if(ind > 0):
                    diff_array[ind] = np.linalg.norm((data-old_data)[2:])/np.linalg.norm(old_data[2:])
                
                var = phase
                if(self.angle):
                    var = angle

                ax.scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=var, cmap=colormap, vmin=vmin, vmax=vmax)

                ax.set_title(f"k={k:.3f} T={T:.3f}")
                ax.set_xlim(-np.pi, np.pi)
                ax.set_ylim(0, np.pi)
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
                ax.set_yticks(np.linspace(0, np.pi, 5), ['0', 'π/4', 'π/2', '3π/4', 'π'])
                ax.invert_yaxis()

        for ax in axs_flat:
            ax.tick_params(axis='both', which='both', labelsize=18)

        ax2.plot(k_array, T_array)

        plt.tight_layout()
        plt.savefig(f'fig/ciliate_periodic_soln_{sim_dir}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()
    
    def create_state(self):
        self.select_sim()
        fil_references_sphpolar = np.zeros((self.nfil,3))
        for i in range(self.nfil):
            fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])

        state = np.zeros(2*self.nfil+2)
        for fi in range(self.nfil):
            state[fi+2] = 0

        np.savetxt('psi.dat', state, newline = " ")

    def mod_state(self):
        sym_file = 'input/states/reserved_states/sym_state.dat'
        dia_file = 'input/states/reserved_states/dia_state.dat'
        output_file = 'input/states/s5d1.dat'

        sym_state = np.loadtxt(sym_file)
        dia_state = np.loadtxt(dia_file)

        nfil = int((len(sym_state)-2)/2)

        sym_state[2:2+nfil] = util.box(sym_state[2:2+nfil], 2*np.pi)
        dia_state[2:2+nfil] = util.box(dia_state[2:2+nfil], 2*np.pi)

        x = (5*sym_state + dia_state)/6.

        np.savetxt(output_file, x, newline = " ")

    def IVPs(self):

        plt.rcParams.update({'font.size': 27})

        # force = False
        # path = "data/ic_hpc_sim/"

        force = False
        path = "data/ic_hpc_sim_free/"

        force = False
        path = 'data/tilt_test//makeup_pattern/'

        # import re
        # def sort_key(s):
        #     # Split the string by the underscore and convert the second part to an integer
        #     return int(s.split('_')[1])
        # folders = sorted(util.list_folders(path), key=sort_key)

        folders = util.list_folders(path)
        print(folders)

        self.plot_end_frame_setting = 60000
        self.frames_setting = 600

        # Extract num_sim from the first folder
        # All folders should have the same num_sim!
        self.dir = path + folders[0] + '/'
        self.read_rules()
        r_data = np.zeros((len(folders), self.num_sim))
        k_data = np.zeros((len(folders), self.num_sim))
        tilt_data = np.zeros((len(folders), self.num_sim))
        avg_vz_data = np.zeros((len(folders), self.num_sim))
        avg_speed_data = np.zeros((len(folders), self.num_sim))
        avg_speed_along_axis_data = np.zeros((len(folders), self.num_sim))
        avg_rot_speed_data = np.zeros((len(folders), self.num_sim))
        avg_rot_speed_along_axis_data = np.zeros((len(folders), self.num_sim))
        eff_data = np.zeros((len(folders), self.num_sim))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for fi, folder in enumerate(folders):
            self.dir = path + folder + '/'
            print(self.dir)
            self.read_rules()

            k_arrays = self.pars_list['spring_factor']
            tilt_arrays = self.pars_list['tilt_angle']
            r_arrays = np.zeros(np.shape(k_arrays))
            avg_vz_arrays = np.zeros(np.shape(k_arrays))
            avg_speed_arrays = np.zeros(np.shape(k_arrays))
            avg_speed_along_axis_arrays = np.zeros(np.shape(k_arrays))
            avg_rot_speed_arrays = np.zeros(np.shape(k_arrays))
            avg_rot_speed_along_axis_arrays = np.zeros(np.shape(k_arrays))
            dis_arrays = np.zeros(np.shape(k_arrays))
            eff_arrays = np.zeros(np.shape(k_arrays))
            
            for ind in range(self.num_sim):
                self.index = ind

                # try:
                self.select_sim()

                body_pos_array = np.zeros((self.frames, 3))
                body_axis_array = np.zeros((self.frames, 3))
                body_q_array = np.zeros((self.frames, 4))

                

                fil_references_sphpolar = np.zeros((self.nfil,3))
                for i in range(self.nfil):
                    fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])

                body_states_f = open(self.simName + '_body_states.dat', "r")
                fil_states_f = open(self.simName + '_true_states.dat', "r")
                if force:
                    seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                    seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                    blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                    blob_references_f = open(self.simName + '_blob_references.dat', "r")

                    blob_references_str = blob_references_f.readline()
                    blob_references= np.array(blob_references_str.split(), dtype=float)
                    blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))

                print(f"[{self.plot_start_frame} - {self.plot_end_frame}]")
                for t in range(self.plot_end_frame):
                    body_states_str = body_states_f.readline()
                    fil_states_str = fil_states_f.readline()
                    if force:
                        seg_forces_str = seg_forces_f.readline()
                        seg_vels_str = seg_vels_f.readline()
                        blob_forces_str = blob_forces_f.readline()
                    if(t>=self.plot_start_frame):
                        body_states = np.array(body_states_str.split()[1:], dtype=float)
                        fil_states = np.array(fil_states_str.split()[2:], dtype=float)

                        phases = fil_states[:self.nfil]
                        r = np.abs(np.sum(np.exp(1j*phases))/self.nfil)
                        r_arrays[ind] += r

                        body_pos_array[t-self.plot_start_frame] = body_states[0:3]
                        body_q_array[t-self.plot_start_frame] = body_states[3:7]

                        R = util.rot_mat(body_states[3:7])
                        body_axis = np.matmul(R, np.array([0,0,1]))
                        body_axis_array[t-self.plot_start_frame] = body_axis

                        if force:
                            seg_forces = np.array(seg_forces_str.split()[1:], dtype=float)
                            seg_vels = np.array(seg_vels_str.split()[1:], dtype=float)
                            blob_forces= np.array(blob_forces_str.split()[1:], dtype=float)
                            # Need to patch this for diffeent output format....
                            if(len(body_vels_str.split())==6):
                                body_vels= np.array(body_vels_str.split(), dtype=float)
                            else:
                                body_vels= np.array(body_vels_str.split()[1:], dtype=float)

                            seg_forces = np.reshape(seg_forces, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                            seg_vels = np.reshape(seg_vels, (int(self.pars['NSEG']*self.pars['NFIL']), 6))
                            blob_forces = np.reshape(blob_forces, (int(self.pars['NBLOB']), 3))
                            body_vels_tile = np.tile(body_vels, (int(self.pars['NBLOB']), 1))
                            blob_vels = body_vels_tile[:, 0:3] + np.cross(body_vels_tile[:, 3:6], blob_references)

                            speed = np.sqrt(np.sum(body_vels[0:3]*body_vels[0:3], 0))
                            dis = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)
                            eff = 6*np.pi*self.radius*speed**2/dis
                            v_arrays[ind] += speed
                            dis_arrays[ind] += dis
                            eff_arrays[ind] += eff
                
                
                body_vel_array = np.diff(body_pos_array, axis=0)/self.dt/self.fillength
                body_vz_array = body_vel_array[:,0]
                body_speed_along_axis_array = np.sum(body_vel_array * body_axis_array[:-1], axis=1)
                body_speed_array = np.linalg.norm(body_vel_array, axis=1)
                body_rot_vel_array = util.compute_angular_velocity(body_q_array, self.dt)
                body_rot_speed_along_axis_array = np.sum(body_rot_vel_array * body_axis_array[:-1], axis=1)
                body_rot_speed_array = np.linalg.norm(body_rot_vel_array, axis=1)

                
                avg_vz_arrays[ind] = np.mean(body_vz_array)
                avg_speed_arrays[ind] = np.mean(body_speed_array)
                avg_speed_along_axis_arrays[ind] = np.mean(body_speed_along_axis_array)
                avg_rot_speed_arrays[ind] = np.mean(body_rot_speed_array)
                avg_rot_speed_along_axis_arrays[ind] = np.mean(body_rot_speed_along_axis_array)
                r_arrays[ind] /= self.frames
                print(avg_speed_arrays[ind])
                if force:
                    
                    dis_arrays[ind] /= self.frames
                    eff_arrays[ind] /= self.frames
            
                # except:
                #     print("Something went wrong")
                #     pass

            r_data[fi] = r_arrays
            k_data[fi] = k_arrays
            tilt_data[fi] = tilt_arrays
            avg_vz_data[fi] = avg_vz_arrays
            avg_speed_data[fi] = avg_speed_arrays
            avg_speed_along_axis_data[fi] = avg_speed_along_axis_arrays
            avg_rot_speed_data[fi] = avg_rot_speed_arrays
            avg_rot_speed_along_axis_data[fi] = avg_rot_speed_along_axis_arrays

            ax.scatter(k_arrays, r_arrays, marker='x', label = folder, c='black')

        # save data
        np.save(f"{path}r_data.npy", r_data)
        np.save(f"{path}k_data.npy", k_data)
        np.save(f"{path}tilt_data.npy", tilt_data)
        np.save(f"{path}avg_vz_data.npy", avg_vz_data)
        np.save(f"{path}avg_speed_data.npy", avg_speed_data)
        np.save(f"{path}avg_speed_along_axis_data.npy", avg_speed_along_axis_data)
        np.save(f"{path}avg_rot_speed_data.npy", avg_rot_speed_data)
        np.save(f"{path}avg_rot_speed_along_axis_data.npy", avg_rot_speed_along_axis_data)
        
        if force:
            np.save(f"{path}eff_data.npy", eff_data)


        ax.set_ylim(0)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$<r>$')
        
        if force:
            ax2.set_ylim(0)
            ax2.set_xlabel(r'$k$')
            ax2.set_ylabel(r'$<v/L>$')

            ax3.set_ylim(0)
            ax3.set_xlabel(r'$k$')
            ax3.set_ylabel(r'$<Efficiency>$')

        # ax.legend()
        # ax2.legend()
        
        fig.tight_layout()
        # fig.savefig(f'fig/IVP_order_parameters_{free_string}.pdf', bbox_inches = 'tight', format='pdf')
        # fig.savefig(f'fig/IVP_order_parameters_{free_string}.png', bbox_inches = 'tight', format='png', transparent=True)

        plt.show()

    def view_bisection(self):
        colormap = 'twilight_shifted'
        # colormap = 'hsv'

        k_string = 'k0.030'
        iteration_string = 'iteration2_1e-7'
        edge_section = f'section13'

        k_string = 'k0.020'
        iteration_string = 'iteration3_1e-7'
        edge_section = f'section16'

        path = f"data/bisection/{k_string}/{edge_section}/{iteration_string}/"

        folders = util.list_folders(path)
        folders.sort()
        print(folders)
        num_sim = len(folders)
        ncol = 4
        nrow = -(-num_sim//ncol)

        self.plot_end_frame_setting = 12000
        self.frames_setting = 100000
        window_size = 1

        fig1, axs1 = plt.subplots(1, num_sim, figsize=(8, 4), sharex=True, sharey=True)
        ax_invi1 = fig1.add_subplot(111, frameon=False)
        axs_flat1 = axs1.ravel()

        fig2, axs2 = plt.subplots(nrow, ncol, figsize=(8, 4), sharex=True, sharey=True)
        axs_flat2 = axs2.ravel()

        axs_flat2[0].set_xlim(-np.pi, np.pi)
        axs_flat2[0].set_ylim(0, np.pi)
        axs_flat2[0].set_xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
        axs_flat2[0].set_yticks(np.linspace(0, np.pi, 3), ['0', 'π/2', 'π'])
        plt.gca().invert_yaxis()

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$r$')
        ax3.set_title(rf'{k_string}')

        axs_flat1[0].set_ylim(0, 1)
        axs_flat1[0].set_xticks([])
        axs_flat1[0].set_yticks(np.linspace(0, 1, 9))
        axs_flat1[0].set_ylabel(r'$<r>$')
        ax_invi1.set_xlabel(r'$t$')
        ax_invi1.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

        alphas = list()
        for folder in folders:
            alphas.append(float(folder.split('_')[1][5:]))
        alpha_gap = alphas[1] - alphas[0]
        
        for fi, ax in enumerate(axs_flat1):
            if (fi < num_sim):
                folder = folders[fi]

                self.dir = path + folder + '/'
                print(self.dir)
                self.read_rules()
                

                # k_arrays = self.pars_list['spring_factor']
                # r_arrays = np.zeros(np.shape(k_arrays))
                # alpha_arrays = np.array([float(folder.split('_')[1][5:])])
                
                for ind in range(self.num_sim):
                    self.index = ind
                    self.select_sim()

                    r_array = np.zeros(self.frames_setting)
                    alpha_array = np.ones(np.shape(r_array))*float(folder.split('_')[1][5:])
                    
                    windowd_length = self.frames - window_size + 1
                    r_avg_array = np.zeros(windowd_length)
                    alpha_avg_array = np.ones(windowd_length)*float(folder.split('_')[1][5:])
                    time_avg_array = np.arange(windowd_length)

                    fil_references_sphpolar = np.zeros((self.nfil,3))
                    for i in range(self.nfil):
                        fil_references_sphpolar[i] = util.cartesian_to_spherical(self.fil_references[3*i: 3*i+3])

                    fil_states_f = open(self.simName + '_true_states.dat', "r")

                    print(f"[{self.plot_start_frame} - {self.plot_end_frame}]")
                    for t in range(self.plot_end_frame):
                        fil_states_str = fil_states_f.readline()
                        if(t>=self.plot_start_frame):
                            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                            fil_states[:self.nfil] = util.box(fil_states[:self.nfil], 2*np.pi)
                            phases = fil_states[:self.nfil]

                            r = np.abs(np.sum(np.exp(1j*phases))/self.nfil)
                            r_array[t-self.plot_start_frame] = r
                        
                        if(t==self.plot_end_frame-1):
                            cmap = mpl.colormaps[colormap]
                            colors = cmap(phases/2/np.pi)

                            x1 = np.insert(fil_states, 0, [ind, t])
                            np.savetxt(f'data/bisection/ini_states/' + f"state{fi+1}.dat", x1, newline = " ")

                            # ax.scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)
                            ax.set_title(r"$\alpha$={:.8f}".format(alphas[fi]) + '\n'*(fi%2==0),  fontsize=10)
                            axs_flat2[fi].set_title(folder.split('_')[0] + '\n' +  folder.split('_')[1][:15])
                            axs_flat2[fi].scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)

                    # r_arrays[ind] /= self.frames
                    for i in range(windowd_length):
                        r_avg_array[i] = np.mean(r_array[i:i+window_size])


                plot_gap = 1
                
                plot_x = time_avg_array[::plot_gap]
                plot_y = r_avg_array[::plot_gap]

                # plot_samp = len(plot_x)
                # step=30
                # N_segments = int(plot_samp/step)
                # [ax.plot(plot_x[step*i:step*(i+1)],\
                #            plot_y[step*i:step*(i+1
                #                                )],\
                #               alpha=np.min([0.1 + i/N_segments,1]),\
                #                 c='black')\
                #                   for i in range(N_segments)]

                ax.plot(plot_x, plot_y, label = folder)
                ax3.plot(plot_x, plot_y, label = folder)


        # ax.set_ylim(0)
        # ax.set_xlabel(r'$\alpha$')
        # ax.set_ylabel(r'$<r>$')
        # ax.legend()

        fig1.tight_layout()
        # fig1.savefig(f'fig/bisection_{k_string}_{iteration_string}.pdf', bbox_inches = 'tight', format='pdf')
        
        fig2.tight_layout()

        fig3.tight_layout()
        fig3.savefig(f'fig/edgestate_{k_string}_{iteration_string}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def footpaths(self):
        free = False
        path = "data/ic_hpc_sim/"
        # path = "data/slow_converge_sims3/"

        free_string = 'held_fixed'
        if free:
            free_string = 'free'

        folders = util.list_folders(path)
        print(folders)

        self.plot_end_frame_setting = 30000
        self.frames_setting = 3000

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)

        for fi, folder in enumerate(folders[3:4]):
            self.dir = path + folder + '/'
            print(self.dir)
            self.read_rules()

            k_arrays = self.pars_list['spring_factor']
            r_arrays = np.zeros(np.shape(k_arrays))
            v_arrays = np.zeros(np.shape(k_arrays))
            dis_arrays = np.zeros(np.shape(k_arrays))
            eff_arrays = np.zeros(np.shape(k_arrays))

            for ind in range(self.num_sim):
                self.index = ind
                # try:
                self.select_sim()

                fil_references_sphpolar = np.zeros((self.nfil,3))
                fil_references_cartersian = np.zeros((self.nfil,3))
                for fil in range(self.nfil):
                    fil_references_sphpolar[fil] = util.cartesian_to_spherical(self.fil_references[3*fil: 3*fil+3])
                    fil_references_cartersian[fil] = self.fil_references[3*fil: 3*fil+3]
                azim_array = fil_references_sphpolar[:,1]
                polar_array = fil_references_sphpolar[:,2]
                sorted_indices = np.argsort(polar_array)

                # Create a mapping to its nearest filament
                from scipy.spatial.distance import cdist
                def nearest_particles(positions):
                    distances = cdist(positions, positions)
                    np.fill_diagonal(distances, np.inf)
                    nearest_indices = np.argmin(distances, axis=1)
                    return nearest_indices
                nearest_indices = nearest_particles(fil_references_cartersian)


                fil_states_f = open(self.simName + '_true_states.dat', "r")
                time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
                r_array = np.zeros(self.frames_setting)
                corr_array = np.zeros(self.frames_setting)

                window_size = 30
                windowd_length = self.frames - window_size + 1
                r_avg_array = np.zeros(windowd_length)
                corr_avg_array = np.zeros(windowd_length)

                if free:
                    seg_forces_f = open(self.simName + '_seg_forces.dat', "r")
                    seg_vels_f = open(self.simName + '_seg_vels.dat', "r")
                    blob_forces_f = open(self.simName + '_blob_forces.dat', "r")
                    blob_references_f = open(self.simName + '_blob_references.dat', "r")
                    body_vels_f = open(self.simName + '_body_vels.dat', "r")

                    blob_references_str = blob_references_f.readline()
                    blob_references= np.array(blob_references_str.split(), dtype=float)
                    blob_references = np.reshape(blob_references, (int(self.pars['NBLOB']), 3))

                print(f"[{self.plot_start_frame} - {self.plot_end_frame}]")
                for t in range(self.plot_end_frame):
                    fil_states_str = fil_states_f.readline()
                    if free:
                        seg_forces_str = seg_forces_f.readline()
                        seg_vels_str = seg_vels_f.readline()
                        blob_forces_str = blob_forces_f.readline()
                        body_vels_str = body_vels_f.readline()
                    if(t>=self.plot_start_frame):
                        fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                        fil_phases = fil_states[:self.nfil]
                        fil_phases = util.box(fil_phases, 2*np.pi)
                        # fil_phases_sorted = fil_phases[sorted_indices]
                        # sin_phases_sorted = np.sin(fil_phases_sorted)

                        # Coordination number 1
                        # phase_diff = np.diff(sin_phases_sorted, prepend=sin_phases_sorted[-1])
                        # corr = np.abs(phase_diff[:-1]) + np.abs(phase_diff[1:])
                        # corr_array[t-self.plot_start_frame] = np.mean(corr)

                        diff = np.sin(fil_phases) - np.sin(fil_phases[nearest_indices])
                        corr_array[t-self.plot_start_frame] = np.mean(np.abs(diff))
                        r_array[t-self.plot_start_frame] = np.abs(np.sum(np.exp(1j*fil_phases))/self.nfil)
            
                for i in range(windowd_length):
                    r_avg_array[i] = np.mean(r_array[i:i+window_size])
                    corr_avg_array[i] = np.mean(corr_array[i:i+window_size])
                ax.plot(time_array[:windowd_length], r_avg_array)
                ax.set_xlabel('t/T')
                ax.set_ylabel('<r>')
                ax.set_xlim(time_array[0], time_array[-1])
                # ax.set_ylim(0)

                ax2.plot(time_array[:windowd_length], corr_avg_array)
                ax2.set_xlabel('t/T')
                ax2.set_ylabel(r'Synchronisation number')

                ax3.plot(corr_avg_array, r_avg_array)
                # ax3.set_xlim(0)
                # ax3.set_ylim(0)
                ax3.set_xlabel(r'Synchronisation number')
                ax3.set_ylabel(r'$<r>$')

                ax4.hexbin(corr_avg_array, r_avg_array, cmap='viridis', gridsize=10)


                # except:
                #     print("Something went wrong")
                #     pass
            
            # ax.scatter(k_arrays, r_arrays, marker='x', label = folder, c='black')
            # ax.scatter(k_arrays, r_arrays, marker='x', label = folder)
            # if free:
            #     ax2.scatter(k_arrays, v_arrays/49.4, marker='x', label = folder, c='black')
            #     ax3.scatter(k_arrays, eff_arrays, marker='x', label = folder, c='black')


        fig.tight_layout()
        # fig.savefig(f'fig/IVP_order_parameters_{free_string}.pdf', bbox_inches = 'tight', format='pdf')
        fig2.tight_layout()
        # fig2.savefig(f'fig/IVP_velocities_{free_string}.pdf', bbox_inches = 'tight', format='pdf')
        fig3.tight_layout()
        # fig3.savefig(f'fig/IVP_efficiencies_{free_string}.pdf', bbox_inches = 'tight', format='pdf')
        plt.show()

    def compute_avg(self):
        k_array = self.pars_list['spring_factor']
        tilt_array = self.pars_list['tilt_angle']
        avg_speed = np.zeros(self.num_sim)
        avg_rot_speed = np.zeros(self.num_sim)
        
        save_dir = f'{self.dir}/avg_quantities/'
        os.system(f'mkdir -p {save_dir}')

        for ind in range(self.num_sim):
            try:
                self.index = ind
                self.select_sim()

                body_states_f = open(self.simName + '_body_states.dat', "r")

                time_array = np.arange(self.plot_start_frame, self.plot_end_frame )/self.period
        
                body_pos_array = np.zeros((len(time_array), 3))
                body_q_array = np.zeros((len(time_array), 4))

                for i in range(self.plot_end_frame):
                    print(" frame ", i, "/", self.plot_end_frame, "          ", end="\r")
                    body_states_str = body_states_f.readline()

                    if(i>=self.plot_start_frame):
                        body_states = np.array(body_states_str.split()[1:], dtype=float)

                        body_pos_array[i-self.plot_start_frame] = body_states[0:3]
                        body_q_array[i-self.plot_start_frame] = body_states[3:7]

                body_vel_array = np.diff(body_pos_array, axis=0)/self.dt
                body_speed_array = np.linalg.norm(body_vel_array, axis=1)
                body_rot_vel_array = util.compute_angular_velocity(body_q_array, self.dt)
                body_rot_speed_array = np.linalg.norm(body_rot_vel_array, axis=1)

                avg_speed[ind] = np.mean(body_speed_array)
                avg_rot_speed[ind] = np.mean(body_rot_speed_array)

                # ax.plot(time_array, body_speed_array/self.fillength, label=f"{self.index}) nblob={self.nblob}")
            except:
                print("WARNING: " + self.simName + " not found.")

        afix = ''
        np.save(f'{save_dir}/k_array{afix}.npy', k_array)
        np.save(f'{save_dir}/tilt_array{afix}.npy', tilt_array)
        np.save(f'{save_dir}/avg_speed{afix}.npy', avg_speed)
        np.save(f'{save_dir}/avg_rot_speed{afix}.npy', avg_rot_speed)
        print(f'index={self.index} avg speed={avg_speed/self.fillength} avg rot speed={avg_rot_speed}')
        

#