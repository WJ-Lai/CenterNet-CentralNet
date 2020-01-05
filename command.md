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

12-22:测试mAP为衡量指标以及加入l2正则+数据增强+模型低配+变小n=3
python main.py --dataset rgb --arch hourglass --batch_size 8 --exp_id mAP --not_rand_crop
python main.py --dataset fir --arch hourglass --batch_size 8 --exp_id mAP --not_rand_crop
python main.py --dataset mir --arch hourglass --batch_size 8 --exp_id mAP --not_rand_crop
python main.py --dataset nir --arch hourglass --batch_size 8 --exp_id mAP --not_rand_crop

python main.py --dataset rgb --arch dla --batch_size 16


12-24:测试mAP为衡量指标+模型压缩通道+lr250降1次
python main.py --dataset rgb --arch hourglass --batch_size 8 --exp_id mAP2
python main.py --dataset fir --arch hourglass --batch_size 8 --exp_id mAP2

12-25:将12-24训练到1000epoch
python main.py --dataset fir --arch hourglass --batch_size 8 --exp_id mAP2 --resume --log_dir /media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Checkpoint/CenterNet-CentralNet/ctdet/mAP2/fir/logs_2019-12-25-03-28/ --num_epochs 1000


