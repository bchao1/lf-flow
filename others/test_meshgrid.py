import torch 
import numpy as np

if __name__ == "__main__":
    h = np.arange(10)
    w = np.arange(10)

    grid_torch = torch.meshgrid(torch.tensor(h), torch.tensor(w))
    grid_numpy = np.meshgrid(h, w)
    print("Torch grid:")
    print(grid_torch[0].numpy())

    print("Numpy grid")
    print(grid_numpy[0])