from lib.opts import opts
import os
opt = opts().parse()

sensor = opt.dataset

# categories = ['bike', 'car', 'car_stop', 'color_cone', 'person']
categories = ['bike', 'car', 'color_cone', 'person']
main_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral'
# images_save_path = os.path.join(main_path, 'exp/output_images/', sensor)
images_save_path = os.path.join('/home/vincent/20191117')
train_size = 512

# rgb
rgb_mean = [0.181065, 0.171860, 0.175805]
rgb_std = [0.275618, 0.261550, 0.266921]
# fir
fir_mean = [0.168333, 0.168333, 0.168333]
fir_std = [0.202114, 0.202114, 0.202114]
# mir
mir_mean = [0.147748, 0.147748, 0.147748]
mir_std = [0.183103, 0.183103, 0.183103]
# nir
nir_mean = [0.187101, 0.187101, 0.187101]
nir_std = [0.271039, 0.271039, 0.271039]

own_dataset_num_classes = len(categories)
default_dataset = sensor

default_mean = {'rgb':rgb_mean, 'fir':fir_mean, 'mir':mir_mean, 'nir':nir_mean}
default_std = {'rgb':rgb_std, 'fir':fir_std, 'mir':mir_std, 'nir':nir_std}

# ctdet_default_num_classes = own_dataset_num_classes
# ctdet_default_mean = default_mean[sensor]
# ctdet_default_std = default_std[sensor]
