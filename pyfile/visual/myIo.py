import os
import pandas as pd
import numpy as np

def read_blob_references(fileName):
    try:
        with open(fileName, 'r') as file:
            lines = file.readlines()
            data = [line.strip().split() for line in lines]
            data = np.array(data, dtype=float)  # Assuming your data is numeric
            return data[:, :].reshape(-1)
    except Exception as e:
        print(f"Error: {e}")
        return []
    # ret = pd.read_csv(fileName, sep = ' ', header=None)
    # return ret.iloc[:,:-1].to_numpy().reshape(-1)

def read_fil_references(fileName):
    try:
        with open(fileName, 'r') as file:
            lines = file.readlines()
            data = [line.strip().split() for line in lines]
            data = np.array(data, dtype=float)  # Assuming your data is numeric
            return data[:, :].reshape(-1)
    except Exception as e:
        print(f"Error: {e}")
        return []
    # ret = pd.read_csv(fileName, sep = ' ', header=None)
    # return ret.iloc[:,:-1].to_numpy().reshape(-1)
    
def read_pars(fileName):
    ret_pardict = {}
    df = pd.read_csv(fileName, sep=' %% ', header=None, engine='python')
    for i in range(len(df)):
        ret_pardict[df.iloc[i, 1]] = df.iloc[i, 0]
    return ret_pardict

def write_line(text, fileName):
    with open(fileName, 'a') as the_file:
        the_file.write(text + '\n')
    
def clean_file(fileName):
    open(fileName, 'w')

def get_boxsize_from_name(filename):
    str_list = filename.split('_')
    try:
        Lx, Ly, Lz = [float(s) for s in str_list[-3:]]
        return Lx, Ly, Lz
    except:
        print("WARNING: Filename not supported for auto boxing.")
        return (float('inf'), float('inf'), float('inf'))
    
def get_ciliate_data_from_name(filename):
    str_list = filename.split('_')
    try:
        R, Tor = float(str_list[-2][:-1]), float(str_list[-1][:-7])
        # for s in str_list[-2:]:
        #     print(s)
        #     if(s.endswith('R')):
        #         R = float(s[:-1])
        #     if(s.endswith('torsion')):
        #         Tor = float(s[:-7])
        return R, Tor
    except:
        print("WARNING: Filename not supported for auto ciliating.")
        print("Error could be incurred by default values.")
        return 5.0, 2.0