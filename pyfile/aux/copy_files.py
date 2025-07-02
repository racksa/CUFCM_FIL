
import os
source_folder = 'data/for_paper/flowfield_example/20250522_flowfield_free/'
destination_folder = 'data/output/flowfield/'

source_folder = "data/tilt_test/makeup_pattern/20240724_symplectic/"
destination_folder = 'data/output/tilt/symplectic/'

# source_folder = "data/for_paper/giant_swimmer_rerun/20250507/"
# destination_folder = 'data/output/big_sphere/'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Number of lines to keep
num_lines = 3000

# The extra prefix to insert
insert_prefix = '_1.0000dp_0.0000noise_0.0000ospread'

# Loop through files
for filename in os.listdir(source_folder):
    if filename.endswith('body_states.dat') or filename.endswith('true_states.dat'):
        source_path = os.path.join(source_folder, filename)

        # Read last num_lines lines
        with open(source_path, 'r') as src_file:
            lines = src_file.readlines()[-num_lines:]

        # Process each line: replace first number
        processed_lines = []
        for idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if parts:
                parts[0] = str(idx * 10)
                processed_lines.append(' '.join(parts) + '\n')

        # Split the filename into:
        #   prefix_part + rest_part
        if filename.endswith('body_states.dat'):
            prefix_part = filename.split('_body_states.dat')[0]
            rest_part = '_body_states.dat'
        elif filename.endswith('true_states.dat'):
            prefix_part = filename.split('_true_states.dat')[0]
            rest_part = '_true_states.dat'
        else:
            prefix_part = filename
            rest_part = ''

        # Create new filename: keep original prefix_part, insert new insert_prefix, keep rest_part
        new_filename = prefix_part + insert_prefix + rest_part

        # Write the processed lines to the new file in destination folder
        destination_path = os.path.join(destination_folder, new_filename)
        with open(destination_path, 'w') as dst_file:
            dst_file.writelines(processed_lines)

        print(f"Processed, copied and renamed: {new_filename}")

print("Done.")

