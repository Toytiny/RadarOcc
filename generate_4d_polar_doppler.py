import os
import scipy.io
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(clip, file, input_dir, output_dir):
    file_path = os.path.join(input_dir, file)
    data = np.load(file_path)
    # data = data.astype(np.float16)
    d_dim, r_dim, a_dim, e_dim = data.shape
     
    
    idx = file.split("_")[-1].split(".")[0]
    # assert np.isinf(data).sum() == 0
    # assert np.isnan(data).sum() == 0
    # output_file_path_cube = os.path.join(output_dir, f"DREA_{idx}.npy")
    # np.save(output_file_path_cube, data.astype(np.float16))
    cube = np.mean(data,axis=0)
    cube_flat = cube.reshape(cube.shape[0], -1)
    k = 250
    top_k_idx = np.argpartition(cube_flat, -k, axis=1)[:, -k:]

    mask_flat = np.zeros_like(cube_flat, dtype=bool)
    mask_flat[np.arange(cube_flat.shape[0])[:, None], top_k_idx] = True
    mask = mask_flat.reshape(cube.shape)

    range_inds = np.arange(cube.shape[0])[:, None]
    elevation_inds = top_k_idx // cube.shape[2]
    azimuth_inds = top_k_idx % cube.shape[2]

    range_inds_flat = np.repeat(range_inds, k).flatten()
    elevation_inds_flat = elevation_inds.flatten()
    azimuth_inds_flat = azimuth_inds.flatten()
    power_val = data[:,range_inds_flat, elevation_inds_flat, azimuth_inds_flat]
    
    top_indices = np.argpartition(power_val, -5, axis=0)[-5:]
    N_dim = power_val.shape[1]
    top_values = power_val[top_indices, np.arange(N_dim)]

    # Step 3: Calculate the mean and variance
    means = np.mean(power_val, axis=0)
    variances = np.var(power_val, axis=0)

    # Step 4: Combine into new array
    new_data = np.vstack((top_values, top_indices, means, variances))

    # Transpose and reshape to get desired format: 8 x N_dim
    new_data = np.vstack((new_data[:5], new_data[5:10], new_data[10], new_data[11])).reshape(12, N_dim)

    output_file_path = os.path.join(output_dir, f"EAsparse_{idx}.npz")
    np.savez(output_file_path, range_ind=range_inds_flat, elevation_ind=elevation_inds_flat, azimuth_ind=azimuth_inds_flat, power_val=new_data)
    return f"{clip} {file} processed successfully"
import gc
def process_clip(clip):
    output_dir = os.path.join(output_base_dir, clip, "radar_tensor_12doppler")
    input_dir = os.path.join(input_base_dir, clip, "radar_polar_cube")
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
                gc.collect()  # Collect garbage after every 100 files

input_base_dir = "/mnt/18T-Data/kradar/"
output_base_dir = "/mnt/18T-Data/kradar/"
clips = os.listdir(input_base_dir),
print(clips)
clips = [clip for clip in clips[0] if clip.isdigit()]
clips = sorted(clips, key=lambda x: int(x))

for clip in clips:
    if int(clip) >= 19:
        process_clip(clip)
