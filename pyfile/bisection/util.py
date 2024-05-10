import numpy as np
import math
import pandas as pd
import os

def box(x, box_size):
    return x - np.floor(x/box_size)*box_size

def cartesian_to_spherical(x):
    """
    Convert Cartesian coordinates to spherical polar coordinates.
    
    Args:
        x (float, float, float): cartesian-coordinate.
    
    Returns:
        tuple: (r, theta, phi), where r is the radial distance, theta is the polar angle (azimuthal angle),
               and phi is the elevation angle (zenith angle).
    """
    r = math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    theta = math.atan2(x[1], x[0])
    phi = math.acos(x[2] / r)
    
    return r, theta, phi

def read_fil_references(fileName):
    ret = pd.read_csv(fileName, sep = ' ', header=None)
    return ret.iloc[:,:-1].to_numpy().reshape(-1)

def list_files_in_directory(directory_path):
    file_list = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            file_list.append(file_path)
    return file_list


def delete_files_in_directory(directory_path):
    try:
        file_list = list_files_in_directory(directory_path)

        if not file_list:
            print(f"No file found in '{directory_path}'. Nothing to delete.")
            return

        print("Files to be deleted:")
        for file_path in file_list:
            print(file_path)

        user_input = input("Do you want to delete these files? (y/n): ")
        if user_input.lower() == 'y':
            for file_path in file_list:
                os.remove(file_path)
            print("All files have been deleted.")
        else:
            print("Deletion canceled. No file was deleted.")
    except Exception as e:
        print(f"Error occurred while deleting files: {e}")

def view_files_in_directory(directory_path):
    try:
        file_list = list_files_in_directory(directory_path)

        if not file_list:
            print(f"No file found in '{directory_path}'.")
            return

        print("Files here:")
        for file_path in file_list:
            print(file_path)

    except Exception as e:
        print(f"Error occurred while viewing files: {e}")