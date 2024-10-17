import numpy as np
from mayavi import mlab
from scipy.interpolate import interp1d
import scipy.io
import os

def plot_tensor(data, frame_index):
    # Define the resolutions
    data  = data[:200]
    range_res = 0.42  # meters
    ele_res = np.deg2rad(1)  # converting degrees to radians
    azi_res = np.deg2rad(1)  # converting degrees to radians

    # Create the spherical coordinate arrays
    r = np.linspace(0, (data.shape[0] - 1) * range_res, data.shape[0])
    ele = np.linspace(-((data.shape[1] - 1) / 2) * ele_res, ((data.shape[1] - 1) / 2) * ele_res, data.shape[1])
    azi = np.linspace(-((data.shape[2] - 1) / 2) * azi_res, ((data.shape[2] - 1) / 2) * azi_res, data.shape[2])

    # Create a 3D grid of spherical coordinates
    R, Ele, Azi = np.meshgrid(r, ele, azi, indexing='ij')

    # Convert spherical coordinates to Cartesian coordinates
    X = R * np.cos(Ele) * np.cos(Azi)
    Y = R * np.cos(Ele) * np.sin(Azi)
    Z = R * np.sin(Ele)

    # Flatten the arrays for plotting
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    r_flat = R.flatten()
    data_flat = data.flatten()
    data_flat = np.log10(data_flat)
    data_flat = np.log10(data_flat)
    # mean = np.mean(data_flat)

# Standardize the data
    # data_flat = (data_flat - mean) 

    # Create a Mayavi figure
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

    # Plot the data using mayavi
    pts = mlab.points3d(x_flat, y_flat, z_flat, data_flat, mode='point', colormap='viridis', scale_factor=100, opacity=0.2)
    # mlab.colorbar(pts, title='Intensity', orientation='vertical')

    # Define the corners in spherical coordinates
    corners = [
        (r[0], ele[0], azi[0]), (r[0], ele[0], azi[-1]), (r[0], ele[-1], azi[0]), (r[0], ele[-1], azi[-1]),
        (r[-1], ele[0], azi[0]), (r[-1], ele[0], azi[-1]), (r[-1], ele[-1], azi[0]), (r[-1], ele[-1], azi[-1])
    ]

    # Convert corners to Cartesian coordinates
    corners_cartesian = [(R * np.cos(E) * np.cos(A), R * np.cos(E) * np.sin(A), R * np.sin(E)) for R, E, A in corners]

    # Extract the x, y, z coordinates of the corners
    x_corners, y_corners, z_corners = zip(*corners_cartesian)

    # Define line connections between corners
    lines = [
        (0, 1), (1, 5), (5, 4), (4, 0),  # Bottom face
        (2, 3), (3, 7), (7, 6), (6, 2),  # Top face
        (0, 2), (1, 3), (4, 6), (5, 7)   # Vertical lines
    ]

    # Plot the curved lines for the longest edges
    def plot_curved_line(start, end, num_points=100):
        r_start, ele_start, azi_start = corners[start]
        r_end, ele_end, azi_end = corners[end]

        r_interp = np.linspace(r_start, r_end, num_points)
        ele_interp = np.linspace(ele_start, ele_end, num_points)
        azi_interp = np.linspace(azi_start, azi_end, num_points)

        x_interp = r_interp * np.cos(ele_interp) * np.cos(azi_interp)
        y_interp = r_interp * np.cos(ele_interp) * np.sin(azi_interp)
        z_interp = r_interp * np.sin(ele_interp)

        mlab.plot3d(x_interp, y_interp, z_interp, tube_radius=0.5, color=(0, 0, 0))

    # Plot the curved lines for the specified edges
    plot_curved_line(5, 4)
    plot_curved_line(6, 7)

    # Plot the straight lines for the remaining edges
    for start, end in lines:
        if (start, end) not in [(5, 4), (7, 6)]:
            mlab.plot3d(
                [x_corners[start], x_corners[end]],
                [y_corners[start], y_corners[end]],
                [z_corners[start], z_corners[end]],
                tube_radius=0.5, color=(0, 0, 0)
            )

    # Set the view
    mlab.view(-180, 150, 250, [53.5, 1.7e-02, 0], -90)  # for cube

    # Save the plot
    save_path = f'/mnt/data/DataSet/viz/4DRT/frame_{frame_index:05d}.png'
    # mlab.show()

    mlab.savefig(save_path)
    mlab.close()

# Load the data
for clip in [54]:
    path = '/mnt/Kradar/K-Radar/{}/radar_tesseract/'.format(clip)
    for i in range(600):
        rt_path = os.path.join(path, 'tesseract_{:05d}.mat'.format(i))
        
        if os.path.exists(rt_path):
            data = scipy.io.loadmat(rt_path)
            data = data['arrDREA']
            data = data.mean(0)

            # Plot and save the data for each frame
            plot_tensor(data, i)
