import PIL.Image as Image
import os


# 定义图像拼接函数
def image_compose(image_path, image_save_path, image_row=2, image_column=2, image_size=[320, 256]):
    to_image = Image.new('RGB', (image_column * image_size[0], image_row * image_size[1]))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, image_row + 1):
        for x in range(1, image_column + 1):
            from_image = Image.open(image_path[image_column * (y - 1) + x - 1]).resize(
                (image_size), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * image_size[0], (y - 1) * image_size[1]))
    image_names = image_path[0]
    image_names = image_names.split('/')[-1]
    image_names = os.path.join(image_save_path, image_names)
    return to_image.save(image_names )  # 保存新图


main_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/output_images/'  # 图片集地址
image_size = [320, 256]  # 每张小图片的大小
image_row = 2  # 图片间隔，也就是合并成一张图后，一共有几行
image_column = 2  # 图片间隔，也就是合并成一张图后，一共有几列
image_save_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/output_images/merge'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
img_name_list = os.listdir(os.path.join(main_path, 'rgb'))
img_name_list.remove('id.txt')
img_name_list.sort()
rgb_path = os.path.join(main_path, 'rgb')
fir_path = os.path.join(main_path, 'fir')
mir_path = os.path.join(main_path, 'mir')
nir_path = os.path.join(main_path, 'nir')
image_path = [rgb_path, nir_path, mir_path, fir_path]

# 得到预测的合并图片
# for img in img_name_list:
#     image_path = [os.path.join(rgb_path, img),
#                   os.path.join(fir_path, img),
#                   os.path.join(mir_path, img),
#                   os.path.join(nir_path, img),]
#     image_compose(image_path, image_save_path=image_save_path,
#                   image_row=image_row, image_column=image_column,
#                   image_size=image_size)  # 调用函数


# 找到vis_ok的原始图片
from ir_to_COCO.extract_night_merge_well import get_img_id
merged_night_id = get_img_id()
vis_ok_path = '/home/vincent/Data/ir_det_dataset/vis_ok'
image_save_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/output_images/compare'  # 图片转换后的地址
img_name_list = os.listdir(vis_ok_path)
img_name_list.remove('.DS_Store')
img_name_list.sort()
pre_save_path = '/home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/output_images/merge'
pre_path_list = os.listdir(pre_save_path)
pre_path_list.sort(key=lambda x:int(x[:-9]))
pre_id = 0
for img_name in img_name_list:
    img_id = img_name.split('.')[0]
    if img_id in merged_night_id:
        ok_path = os.path.join(vis_ok_path, img_name)
        pre_path = os.path.join(pre_save_path, pre_path_list[pre_id])
        image_merge_path = [pre_path, ok_path]
        image_compose(image_merge_path, image_save_path=image_save_path,
                  image_row=1, image_column=2,
                  image_size=[640, 500])  # 调用函数
        print(image_merge_path)
        pre_id += 1
print('')