import torch
import numpy as np

def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    projections = []
    for activations in activation_batch:
        # reshaped_activations = (activations).reshape(
        #     activations.shape[0], -1).transpose()
        reshaped_activations = torch.transpose(torch.reshape(activations, (activations.shape[0], -1)), 0, 1)
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        # U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        U, S, VT = torch.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return torch.stack(projections)
'''

def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)
'''