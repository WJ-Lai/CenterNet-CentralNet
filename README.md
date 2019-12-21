修改内容：

1. data地址、checkpoint地址外移动，若想要制定，则用`--data_dir`、`--exp_dir`
2. 选择传感器种类，如`--fuse rgb,fir,mir
3. 可以在tensorboardX中记录不同类的loss,main_fusion.py可输出不同类的损失（但是总和大于hm_loss有点问题）
4. 可以在tensorboardX中记录权重变化
5. src/test_code中为自己验证代码功能的测试代码
6. 如果想要draw的话，需要在basr_trainer.py的ModleWithLoss_fusion或ModleWithLoss中取消注释，同时model.py中改为model_factory['hourglassdrawing']
7. test_nms中可查看单传感器与nms后融合结果,以及任意传感器组合
8. main_fusion_test_loader.py中为统一场景输入的融合loader的代码，但是没有结局的mead和std的问题(在opt)中，debug函数、detector类会用到
9. large_hourglass_fusion_nms.py实现任意数量的传感器nms后融合
10. mAP作为best model指标
11. 在hourglassd第207行中加入dropout
12. 在Adam中加入l2 norm(weight_decay=5e-4)

test的时候要注意input size

测试集-last model mAP：
rgb:0.476
fir:0.793
mir:0.782
nir:0.706

rgb+fir:0.819
rgb+mir:0.773
fgb+nir:0.733
fir+mir:0.877
fir+nir:0.882
mir+nir:0.894

rgb+fir+mir:0.864
rgb+fir+nir:0.879
rgb+mir+nir:0.865
fir+mir+nir:0.914

rgb+fir+mir+nir:0.876

单传感器：(用统一的exp_id,注意batch_size)
初次训练必须输入的参数:main.py
--dataset
--arch
必须关闭trainval
例如：python main.py --dataset rgb --arch hourglass

继续训练必须输入的参数:main.py
--dataset
--arch
--resume
--log_dir
必须关闭trainval
例如：python main.py --dataset rgb --arch hourglass --resume --log_dir /media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Checkpoint/CenterNet-CentralNet/ctdet/default/logs_2019-12-20-21-20


测试必须输入的参数:test.py and test_nms.py
--dataset
--arch
--load_model
--test_dataset(默认为test)
例如：
python test.py --dataset rgb --arch hourglass --load_model /media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Checkpoint/CenterNet-CentralNet/ctdet/single2/rgb/model_last.pth


