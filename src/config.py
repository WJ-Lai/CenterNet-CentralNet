from lib.opts import opts
import os
opt = opts().parse()

sensor = opt.dataset
print('################## Dataset about %s ##################' % sensor)

# categories = ['bike', 'car', 'car_stop', 'color_cone', 'person']
categories = ['bike', 'car', 'color_cone', 'person']
main_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral'
# images_save_path = os.path.join(main_path, 'exp/output_images/', sensor)
images_save_path = os.path.join('/home/vincent/20191117')
train_size = 512

# rgb
rgb_mean = [0.083381, 0.083511, 0.090018]
rgb_std = [0.165419, 0.165078, 0.174498]
# fir
fir_mean = [0.173141, 0.173141, 0.173141]
fir_std = [0.203299, 0.203299, 0.203299]
# mir
mir_mean = [0.153849, 0.153849, 0.153849]
mir_std = [0.195073, 0.195073, 0.195073]
# nir
nir_mean = [0.115349, 0.115349, 0.115349]
nir_std = [0.174036, 0.174036, 0.174036]

own_dataset_num_classes = len(categories)
default_dataset = sensor

default_mean = {'rgb':rgb_mean, 'fir':fir_mean, 'mir':mir_mean, 'nir':nir_mean}
default_std = {'rgb':rgb_std, 'fir':fir_std, 'mir':mir_std, 'nir':nir_std}

# ctdet_default_num_classes = own_dataset_num_classes
# ctdet_default_mean = default_mean[sensor]
# ctdet_default_std = default_std[sensor]
