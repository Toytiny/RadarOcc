import os
import scipy.io
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to process each file within a clip
def process_file(clip, file, input_dir, output_dir):
    file_path = os.path.join(input_dir, file)
    mat = scipy.io.loadmat(file_path)
    data = torch.tensor(np.log10(mat['arrDREA']).astype(np.float32))
    cube = data.mean(0)
    thr_vale = cube.quantile(0.8)
    range_ind, elevation_ind, azimuth_ind = torch.where(cube > thr_vale)
    power_val = data[:,range_ind, elevation_ind, azimuth_ind]
    range_ind, elevation_ind, azimuth_ind = np.array(range_ind),np.array(elevation_ind),np.array(azimuth_ind)
    
    
    idx = file.split("_")[-1].split(".")[0]
    
    output_file_path = os.path.join(output_dir, f"Sparse_{idx}.npz")
    np.savez(output_file_path, range_ind=range_ind, elevation_ind=elevation_ind, azimuth_ind=azimuth_ind, power_val=power_val)
    return f"{clip} {file} processed successfully"

# Function to handle the processing of a single clip, parallelizing file operations
def process_clip(clip):
    output_dir = os.path.join(output_base_dir, clip, "radar_polar_sparse")
    input_dir = os.path.join(input_base_dir, clip, "radar_tesseract")

    os.makedirs(output_dir, exist_ok=True)
    file_names = sorted(os.listdir(input_dir))

    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submitting file processing tasks for the current clip
        future_to_file = {executor.submit(process_file, clip, file, input_dir, output_dir): file for file in file_names}
        
        # As each file processing completes, you can add logging or error handling here
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'File {file} in clip {clip} generated an exception: {exc}')
            else:
                print(result)

input_base_dir = "/mnt/Kradar/K-Radar/"
output_base_dir = "/mnt/18T-Data/kradar/"
clips = sorted(os.listdir(input_base_dir), key=lambda x: int(x))

# Processing each clip, but within each clip, files are processed in parallel
for clip in clips:
    # if clip == '1':
    #     continue
    process_clip(clip)