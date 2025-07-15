import os
import sys

def copy_file(src, dst, buffer_size=1048576):  # 1MB buffer
    """Copy a file from src to dst without using shutil"""
    try:
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                while True:
                    buf = fsrc.read(buffer_size)
                    if not buf:
                        break
                    fdst.write(buf)
        # Preserve file permissions
        stat = os.stat(src)
        os.chmod(dst, stat.st_mode)
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}", file=sys.stderr)
        return False

def copy_specific_files(source_folder, destination_folder):
    """Recursively copy files and subfolders with specific prefixes"""
    try:
        # Verify source folder
        if not os.path.isdir(source_folder):
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

        # Create destination root folder if needed
        os.makedirs(destination_folder, exist_ok=True)

        # Check permissions
        if not os.access(source_folder, os.R_OK):
            raise PermissionError(f"No read access to: {source_folder}")
        if not os.access(destination_folder, os.W_OK):
            raise PermissionError(f"No write access to: {destination_folder}")

        # Initialize counters
        total_copied = 0
        total_skipped = 0
        prefixes = ('psi', 'bodystate', 'rule')

        # Walk through directory tree
        for root, dirs, files in os.walk(source_folder):
            # Create relative path and corresponding destination directory
            rel_path = os.path.relpath(root, source_folder)
            dest_dir = os.path.join(destination_folder, rel_path)
            
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except PermissionError:
                print(f"Permission denied creating directory: {dest_dir}")
                continue

            # Process files in current directory
            for filename in files:
                if any(filename.startswith(p) for p in prefixes):
                    src_path = os.path.join(root, filename)
                    dst_path = os.path.join(dest_dir, filename)
                    
                    if copy_file(src_path, dst_path):
                        print(f"Copied: {os.path.join(rel_path, filename)}")
                        total_copied += 1
                    else:
                        total_skipped += 1
                else:
                    total_skipped += 1

        print(f"\nDone. Successfully copied {total_copied} files. Skipped {total_skipped} files.")

    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Example usage - replace with your actual paths
    folder_a = "data/tilt_test/makeup_pattern_with_force"
    folder_b = "data/for_paper/IVP_tilt"

    # folder_a = "data/ic_hpc_sim_free_with_force3"
    # folder_b = "data/for_paper/IVP_free"

    # folder_a = "data/ic_hpc_sim_rerun"
    # folder_b = "data/for_paper/IVP_fixed"
    
    copy_specific_files(folder_a, folder_b)