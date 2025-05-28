import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from scipy.io import loadmat
def process_file(clip, file, input_dir, output_dir):
    file_path = os.path.join(input_dir, file)
    mat_data = loadmat(file_path)
    data = mat_data['arrDREA']
    d_dim, r_dim, a_dim, e_dim = data.shape

    idx = file.split("_")[-1].split(".")[0]

    cube = np.mean(data, axis=0)
    cube = torch.tensor(cube)
    thr_vale = cube.quantile(0.9)
    range_ind, elevation_ind, azimuth_ind = torch.where(cube > thr_vale)
    range_ind, elevation_ind, azimuth_ind = np.array(range_ind),np.array(elevation_ind),np.array(azimuth_ind)

    power_val = data[:,range_ind, elevation_ind, azimuth_ind]
    
    top_indices = np.argpartition(power_val, -3, axis=0)[-3:]
    N_dim = power_val.shape[1]
    top_values = power_val[top_indices, np.arange(N_dim)]

    # Step 3: Calculate the mean and variance
    means = np.mean(power_val, axis=0)
    variances = np.var(power_val, axis=0)

    # Step 4: Combine into new array
    new_data = np.vstack((top_values, top_indices, means, variances))

    # Transpose and reshape to get desired format: 8 x N_dim
    new_data = np.vstack((new_data[:3], new_data[3:6], new_data[6], new_data[7])).reshape(8, N_dim)

    output_file_path = os.path.join(output_dir, f"EAsparse_{idx}.npz")
    np.savez(output_file_path, range_ind=range_ind, elevation_ind=elevation_ind, azimuth_ind=azimuth_ind, power_val=new_data)
    return f"{clip} {file} processed successfully"

import gc

def process_clip(clip):
    output_dir = os.path.join(output_base_dir, clip, "radar_tensor_percentile")
    input_dir = os.path.join(input_base_dir, clip, "radar_tesseract")
    os.makedirs(output_dir, exist_ok=True)
    file_names = sorted(os.listdir(input_dir))

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(process_file, clip, file, input_dir, output_dir): file for file in file_names}
        count = 0
        for future in as_completed(future_to_file):
            count += 1
            file = future_to_file[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'File {file} in clip {clip} generated an exception: {exc}')
            else:
                print(result)
            if count % 64 == 0:
                gc.collect()  # Collect garbage after every 64 files

input_base_dir = "/mnt/18T-Data/kradar/"
output_base_dir = "/mnt/18T-Data/kradar/"
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
clips = sorted([d for d in os.listdir(input_base_dir) if is_number(d)], key=lambda x: int(x))

for clip in clips:
    process_clip(clip)
