修改内容：

1. data地址、checkpoint地址外移动，若想要制定，则用`--data_dir`、`--exp_dir`
2. 选择传感器种类，如`--fuse rgb,fir,mir
3. 可以在tensorboardX中记录不同类的loss,main_fusion.py可输出不同类的损失（但是总和大于hm_loss有点问题）
4. 可以在tensorboardX中记录权重变化
5. src/test_code中为自己验证代码功能的测试代码
6. 如果想要draw的话，需要在basr_trainer.py的ModleWithLoss_fusion或ModleWithLoss中取消注释，同时model.py中改为model_factory['hourglassdrawing']