python main_fusion.py --dataset rgb,fir --exp_id rgbfir
python main_fusion.py --dataset rgb,mir --exp_id rgbmir
python main_fusion.py --dataset rgb,nir --exp_id rgbnir

必须设置dataset、arch、train_size

继续训练：
python main_fusion.py --dataset mir,nir --exp_id mirnir --load_model /home/vincent/Checkpoint/CenterNet-CentralNet/ctdet/rgbfir/model_300.pth --log_dir /home/vincent/Checkpoint/CenterNet-CentralNet/ctdet/mirnir/logs_2019-12-03-20-19 --resume

test:
python test_fusion.py --dataset rgb,fir --load_model /home/vincent/Checkpoint/CenterNet-CentralNet/ctdet/rgbfir/model_400.pth


单传感器：
测试：
python test.py --arch hourglass --dataset rgb --load_model /home/vincent/Checkpoint/CenterNet-CentralNet/ctdet/single2/rgb/model_last.pth
