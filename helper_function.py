import os
import random
import shutil
def count_files_in_directory(directory_path):
    # List all entries in the directory
    entries = os.listdir(directory_path)
    # Filter entries to include only files
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
    return len(files)


def split_files_three(source_dir, dest_dir1, dest_dir2, dest_dir3, split_ratio):
    # Ensure destination directories exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
    os.makedirs(dest_dir3, exist_ok=True)

    # Get all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Shuffle the files randomly to ensure a random split
    random.shuffle(files)

    # Determine the split points
    split_point1 = int(split_ratio[0] * len(files))
    split_point2 = int(split_ratio[1] * len(files)) + split_point1

    # Split files into three lists
    group1 = files[:split_point1]
    group2 = files[split_point1:split_point2]
    group3 = files[split_point2:]

    # Move files to the respective directories
    for file in group1:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir1, file))
    
    for file in group2:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir2, file))
    
    for file in group3:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir3, file))

    print(f"Files split successfully: {len(group1)} files in {dest_dir1}, {len(group2)} files in {dest_dir2}, {len(group3)} files in {dest_dir3}.")