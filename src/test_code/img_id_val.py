import os

path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/data/rgb/images'
path_rgb = os.listdir(path)

path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/data/fir/images'
path_fir = os.listdir(path)

path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/data/mir/images'
path_mir = os.listdir(path)

path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/data/nir/images'
path_nir = os.listdir(path)

print(path_rgb==path_fir)
print(path_rgb==path_mir)
print(path_rgb==path_nir)