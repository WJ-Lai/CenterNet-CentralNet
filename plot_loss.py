import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_curve(log_file):
    loss_data = open(log_file)
    all_lines = loss_data.readlines()
    # print(all_lines[4].split(' '))
    # losses
    total_loss = []  # 4
    hm_loss = []  # 7
    wh_loss = []  # 10
    off_loss = []  # 13
    val_loss = []  # 19
    spend_time = []  # 16
    num_lines = len(all_lines)
    for line in range(num_lines):
        total_loss1 = all_lines[line].split(' ')[4]
        hm_loss1 = all_lines[line].split(' ')[7]
        wh_loss1 = all_lines[line].split(' ')[10]
        off_loss1 = all_lines[line].split(' ')[13]
        spend_time1 = all_lines[line].split(' ')[16]

        total_loss.append(float(total_loss1))
        hm_loss.append(float(hm_loss1))
        wh_loss.append(float(wh_loss1))
        off_loss.append(float(off_loss1))
        spend_time.append(float(spend_time1))

    index_val = np.linspace(0, len(all_lines), num=(int((len(all_lines))/5+1))) - 1
    index_val = np.delete(index_val, 0, 0)

    for id in index_val:
        val_loss1 = all_lines[int(id)].split(' ')[19]
        val_loss.append(float(val_loss1))
    return val_loss, total_loss


if __name__ == '__main__':
    # 标准图形绘制
    # sns.set()
    rgb_log_dir = 'logs_2019-11-16-23-03'
    # fir_log_dir = 'logs_2019-10-21-15-19'
    # mir_log_dir = 'logs_2019-10-21-22-34'
    # nir_log_dir = 'logs_2019-10-22-05-50'

    sensor_loss = {'rgbfir': rgb_log_dir}#, 'fir': fir_log_dir, 'mir': mir_log_dir, 'nir': nir_log_dir}

    for sensor, loss in sensor_loss.items():
        if sensor=='rgb':
            continue
        print(sensor)
        main_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla'
        loss_path = os.path.join(main_path, sensor, loss, 'log.txt')
        vloss_res18, loss_res18 = plot_loss_curve(loss_path)  # 读取训练时生成的日志文件
        # vloss_resdcn18, loss_resdcn18 = plot_loss_curve('logresdcn18.txt')
        # vloss_dla, loss_dla = plot_loss_curve('logdla34.txt')
        # vloss_res101, loss_res101 = plot_loss_curve('logres101.txt')
        # vloss_dla34p, loss_dla34p = plot_loss_curve('logdla34p.txt')
        # vloss_hg, loss_hg = plot_loss_curve('loghourglass.txt')

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(range(len(loss_res18)), loss_res18, 'c', label=sensor, linewidth=2)  # 这个label是图线自己的标签；
        # ax.plot(range(len(loss_resdcn18)), loss_resdcn18, 'y', label='resdcn_18_train_loss', linewidth=1)
        # ax.plot(range(len(loss_dla)), loss_dla, 'b', label='dla_34_train_loss', linewidth=1)
        # ax.plot(range(len(loss_res101)), loss_res101, 'g', label='res_101_train_loss', linewidth=1)
        # ax.plot(range(len(loss_dla34p)), loss_dla34p, 'r', label='dla_34_train_loss', linewidth=1)
        # ax.plot(range(len(loss_hg)), loss_hg, 'r', label='hourglass_train_loss', linewidth=1)

        # ax.plot(index_val+1, val_loss, label='val_loss')
        # ax.plot(np.random.randn(1000).cumsum(), label='line2')
        # ax.set_xlim([0, 800])                                      # 设置刻度；
        # ax.set_xticks(range(0, 500, 100))                          # 设置显示的刻度；
        # ax.set_yticklabels(['jan', 'feb', 'mar'])                  # 设置刻度标签；

        # 设置字体大小
        font_ax = {'weight': 'normal', 'size': 15}
        font_title = {'weight': 'normal', 'size': 17}
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        ax.set_xlabel('Epochs', font_ax)  # 设置坐标轴标签；
        ax.set_ylabel('Loss Value', font_ax)
        # ax.text(8750, 20, "海拔", color='red')                     # 加入文本
        # ax.set_title('Loss of CenterNet ( %s )' % sensor, font_title)
        ax.legend(loc='best', fontsize=18)  # 将图例摆放在不遮挡图线的位置即可
        ax.grid()  # 添加网格

        img_name = main_path+'/loss_img/' + sensor + '_val_loss_of_CenterNet.png'
        # plt.savefig(img_name)  # 保存文件到指定文件夹
        plt.show()

        # fig1 = plt.figure(figsize=(12, 8))
        # ax1 = fig1.add_subplot(111)
        # ax1.plot(range(len(vloss_res18)), vloss_res18, 'c', label=sensor, linewidth=2)  # 这个label是图线自己的标签；
        # # ax1.plot(range(len(vloss_resdcn18)), vloss_resdcn18, 'y', label='resdcn_18_val_loss', linewidth=2)
        # # ax1.plot(range(len(vloss_dla)), vloss_dla, 'b', label='dla_34_val_loss', linewidth=2)
        # # ax1.plot(range(len(vloss_res101)), vloss_res101, 'g', label='res_101_val_loss', linewidth=2)
        # # ax1.plot(range(len(vloss_dla34p)), vloss_dla34p, 'r', label='dla_34_val_loss_p', linewidth=2)
        # # ax.plot(index_val+1, val_loss, label='val_loss')
        # # ax.plot(np.random.randn(1000).cumsum(), label='line2')
        # # ax.set_xlim([0, 800])                                      # 设置刻度；
        # # ax.set_xticks(range(0, 500, 100))                          # 设置显示的刻度；
        # # ax.set_yticklabels(['jan', 'feb', 'mar'])                  # 设置刻度标签；
        # ax1.set_xlabel('epochs')  # 设置坐标轴标签；
        # ax1.set_ylabel('loss_value')
        # # ax.text(8750, 20, "海拔", color='red')                     # 加入文本
        # ax1.set_title('val_loss_of_CenterNet')
        # ax1.legend(loc='best')  # 将图例摆放在不遮挡图线的位置即可
        # ax1.grid()  # 添加网格
        # img_name = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/loss_img/'+sensor+'_val_loss_of_CenterNet.png'
        # plt.savefig(img_name)  # 保存文件到指定文件夹
        # plt.show()