import numpy as np
import scipy.io

def generate_sample_mask(NSize, R, NFullSample):
    Nx, Ny = NSize
    mask_x = np.ones(Nx)
    mask_y = np.zeros(Ny)
    Nc = NFullSample
    mask_y[(Ny - Nc) // 2 : ((Ny - Nc) // 2 + Nc)] = 1
    mu = (Ny - 1) / 2
    sigma = (Ny - 1) / 6
    count = 1

    while count <= (Ny // R) - Nc:
        ind_u = int(np.round(np.random.normal(mu, sigma)))
        if 0 <= ind_u <= Ny-1:
            if mask_y[ind_u] == 0:
                count += 1
                mask_y[ind_u] = 1

    sampleMask = np.outer(mask_x, mask_y)
    return sampleMask


x = generate_sample_mask([384,384], 5, 25)  # 4,40
scipy.io.savemat('/mnt/sda1/dl/Mask/384mask_0.2.mat', {'x': x})
print('done')