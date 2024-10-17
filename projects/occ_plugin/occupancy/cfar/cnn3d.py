import torch
import torch.nn as nn
from mmdet.models import NECKS

@NECKS.register_module()
class Simple3DCNN(nn.Module):
    def __init__(self, kernel_size, quantile_rate):
        super(Simple3DCNN, self).__init__()
        padding = (kernel_size - 1) // 2  # Calculate padding
        self.conv = nn.Conv3d(4, 1, kernel_size, stride=1, padding=padding)
        self.quantile_rate = quantile_rate

    def forward(self, arr_cube):
        grid_size = 0.4
        z_min, z_max =-5, 2.6
        y_min, y_max = -51.2, 50.8
        x_min, x_max = -51.2, 50.8

        device=arr_cube.device
        batch_size, dim1, dim2, dim3 = arr_cube.shape

        dim0_indices = torch.arange(dim1,device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(batch_size, -1, dim2, dim3)
        z = dim0_indices * grid_size + z_min
        dim1_indices = torch.arange(dim2,device=device).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(batch_size, dim1, -1, dim3)
        y = dim1_indices * grid_size + y_min
        dim2_indices = torch.arange(dim3,device=device).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_size, dim1, dim2, -1)
        x = dim2_indices * grid_size + x_min
        cube = torch.stack([arr_cube, z, y, x], dim=1)

        print(cube.shape)
        cube = cube.to(torch.float32)
        cube = self.conv(cube)
        cube = cube.squeeze(1)
        cube_sigmoid = torch.sigmoid(cube)
        ###compute loss###

        print(cube.shape)
        z_ind_list = []
        y_ind_list = []
        x_ind_list = []
        for i in range(cube.shape[0]):
    # Compute the quantile for each batch separately
            quantile_value = arr_cube[i].quantile(0.9)
    
    # Find indices where the condition is met for each batch
            z_ind, y_ind, x_ind = torch.where(arr_cube[i] > quantile_value)
    
    # Store the indices for each batch
            z_ind_list.append(z_ind)
            y_ind_list.append(y_ind)
            x_ind_list.append(x_ind)


        return  cube_sigmoid,torch.cat(z_ind_list, dim=0), torch.cat(y_ind_list, dim=0), torch.cat(x_ind_list, dim=0)