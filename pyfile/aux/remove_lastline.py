import os

def remove_last_line(file_path):
    try:
        # Read all lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Remove the last line
        lines = lines[:-1]

        # Write the remaining lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"Last line removed from '{file_path}' successfully.")
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print("An error occurred:", e)


def append_file_contents(file_a, file_b):
    try:
        # Open file A in append mode and file B in read mode
        with open(file_a, 'a') as file_a_handle, open(file_b, 'r') as file_b_handle:
            # Read content of file B
            content_b = file_b_handle.read()
            
            # Append content of file B to file A
            file_a_handle.write(content_b)
            
        print(f"Content of '{file_b}' appended to '{file_a}' successfully.")
    
    except FileNotFoundError:
        print("One or both of the files could not be found.")
    except Exception as e:
        print("An error occurred:", e)


# Example usage
path = f"data/tilt_test/20240710_free_test"

def list_files_with_suffix(directory, suffixes):
    # Ensure suffixes is a tuple to work with str.endswith
    if isinstance(suffixes, str):
        suffixes = (suffixes,)
    else:
        suffixes = tuple(suffixes)
        
    return [f for f in os.listdir(directory) if f.endswith(suffixes)]

# Example usage
suffixes = ('_body_states.dat', '_seg_states.dat', '_true_states.dat')  # List of suffixes
files = list_files_with_suffix(path, suffixes)

print(len(files))

for file_name in files:
    print(path + '/' + file_name)
    remove_last_line(path + '/' + file_name)

# file_a_prefix = f"data/IVP159/20240608/"
# file_b_prefix = f"data/IVP159_continue/20240608_continue1/"
# file_prefix = "ciliate_159fil_9000blob_8.00R_0.0830torsion"

# file_a = f"{file_a_prefix}{file_prefix}_body_states.dat"
# file_b = f"{file_b_prefix}{file_prefix}_body_states.dat"
# remove_last_line(file_a)
# append_file_contents(file_a, file_b)

# file_a = f"{file_a_prefix}{file_prefix}_seg_states.dat"
# file_b = f"{file_b_prefix}{file_prefix}_seg_states.dat"
# remove_last_line(file_a)
# append_file_contents(file_a, file_b)

# file_a = f"{file_a_prefix}{file_prefix}_true_states.dat"
# file_b = f"{file_b_prefix}{file_prefix}_true_states.dat"
# remove_last_line(file_a)
# append_file_contents(file_a, file_b)
