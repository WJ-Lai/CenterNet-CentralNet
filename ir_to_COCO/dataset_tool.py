import os
import shutil
import sys
import json
import glob
import xml.etree.ElementTree as ET
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import shutil


class dataset_tool():

    def __init__(self, ir_det_dataset_path):

        # var about get id
        self.target_img = 'ConvertedImages'
        self.target_ann = 'Annotations_ConvertedSummarized'
        self.dir_name = [self.target_img, self.target_ann, 'vis_ok']
        self.ir_det_dataset_path = ir_det_dataset_path

        # var about output coco format file
        self.sensor_type = ['rgb', 'fir', 'mir', 'nir']
        # image_size = {'rgb':[640, 480], 'fir':[640, 480], 'mir':[320, 256], 'nir':[320, 256]}
        # if ConvertedImages
        self.image_size = {'rgb': [320, 256], 'fir': [320, 256], 'mir': [320, 256], 'nir': [320, 256]}

        self.START_BOUNDING_BOX_ID = 0
        self.PRE_DEFINE_CATEGORIES = {"bike": 0, "car": 1, "car_stop": 2, "color_cone": 3, "person": 4}


        self.dataset_path, self.dataset_coco_path = self.get_dataset_path()
        self.id_list = self.get_id()
        self.dataset_id = self.split_train_test()

    def get_dataset_path(self):
        # get image and ann path
        dataset_path, ir_det_dataset_coco_path = {}, {}
        for sensor in self.sensor_type:
            ir_det_dataset_coco_path[sensor] = {}
        for dir in self.dir_name:
            dataset_path[dir] = \
                os.path.join(ir_det_dataset_path, dir)

        # get coco image and ann path
        coco_main_path = os.path.join(self.ir_det_dataset_path, 'ir_det_dataset_COCO')
        coco_dir_name_list = ['images', 'annotations', 'xml/xml_test', 'xml/xml_train']
        for sensor in self.sensor_type:
            for coco_dir_name in coco_dir_name_list:
                ir_det_dataset_coco_path[sensor][coco_dir_name] = os.path.join(coco_main_path, sensor, coco_dir_name)

        return dataset_path, ir_det_dataset_coco_path

    def get_id(self):
        # return 2999 image id
        id_list, rgb_path = [], os.path.join(self.dataset_path[self.target_img],'rgb')
        img_id_list = os.listdir(rgb_path)
        for img_id in img_id_list:
            id, img_format = img_id.split('.')
            assert img_format=='png', 'It is not a png image'
            id_list.append(id)
        print('Get {} images id.'.format(len(id_list)))
        return id_list

    def get_correct_converted_image_id(self):
        vis_ok_id, vis_ok_id_list = [], os.listdir(self.dataset_path['vis_ok'])
        for vis_ok in vis_ok_id_list:
            id, img_format = vis_ok.split('.')
            assert img_format=='png', 'It is not a png image'
            vis_ok_id.append(id)
        return vis_ok_id

    def split_train_test(self):
        # incorrect converted image is use for train (1553)
        # correct converted image is use for test (1553)
        train_id, test_id, dataset_id = self.id_list, [], {}
        vis_ok_id = self.get_correct_converted_image_id()
        for id in vis_ok_id:
            train_id.remove(id)
        dataset_id['train'], dataset_id['test'] = train_id, vis_ok_id
        print('Get {} train images id.'.format(len(dataset_id['train'])))
        print('Get {} test images id.'.format(len(dataset_id['test'])))
        return dataset_id

    def create_coco_dir(self):
        try:
            shutil.rmtree(os.path.join(self.ir_det_dataset_path, 'ir_det_dataset_COCO'))
        except:
            print('Create folder...')

        for sensor, folder_list in self.dataset_coco_path.items():
            for folder_name, folder in folder_list.items():
                if sensor in ['fir', 'mir', 'nir'] and folder_name in ['annotations', 'xml/xml_test', 'xml/xml_train']:
                    continue
                self.create_dir(folder)

    def mkdir(self, path):
        path=path.strip()
        path=path.rstrip("\\")
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path+' ----- folder created')
            return True
        else:
            print(path+' ----- folder existed')
            return False

    def create_dir(self, dir_name):
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        else:
            self.mkdir(dir_name)

    def get_image(self):
        for sensor in self.sensor_type:
            coco_img_path = self.dataset_coco_path[sensor]['images']
            for img_name in os.listdir(os.path.join(self.dataset_path[self.target_img], sensor)):
                orig_img_path = os.path.join(self.dataset_path[self.target_img], sensor, img_name)
                shutil.copy(orig_img_path, coco_img_path)
        print('Finish get image!')

    def get_xml(self):
        def get_fileID(xml_file):
            filename = xml_file.split('/')[-1]
            filename = filename.split('.')[0]
            return filename

        for use_type in ['train', 'test']:
            for ann_name in os.listdir(self.dataset_path[self.target_ann]):
                ann_id = get_fileID(ann_name)
                if ann_id in self.dataset_id[use_type]:
                    ann_path = os.path.join(self.dataset_path[self.target_ann], ann_name)
                    shutil.copy(ann_path, self.dataset_coco_path['rgb']['xml/xml_'+use_type])
            for sensor in ['fir', 'mir', 'nir']:
                shutil.copytree(self.dataset_coco_path['rgb']['xml/xml_'+use_type], \
                                self.dataset_coco_path[sensor]['xml/xml_' + use_type])
        print('Finish get xml ann!')

    def create_json(self):

        def get_filename(xml_file, voc_images):
            filename = xml_file.split('/')[-1]
            filename = filename.replace('.xml', '.png')
            all_file = os.listdir(voc_images)
            if filename in all_file:
                image_not_exit = False
            else:
                image_not_exit = True
            return filename, image_not_exit

        def get_filename_as_int(filename):
            try:
                filename = os.path.splitext(os.path.basename(filename))[0]
                filename = re.sub("[^0-9]", "", filename)
                return int(filename)
            except:
                raise ValueError("Filename %s is supposed to be an integer." % (filename))

        def get_categories(xml_files):
            """Generate category name to id mapping from a list of xml files.

            Arguments:
                xml_files {list} -- A list of xml file paths.

            Returns:
                dict -- category name to id mapping.
            """
            classes_names = []
            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for member in root.findall("object"):
                    classes_names.append(member[0].text)
            classes_names = list(set(classes_names))
            classes_names.sort()
            return {name: i for i, name in enumerate(classes_names)}

        def get(root, name):
            vars = root.findall(name)
            return vars

        def get_and_check(root, name, length):
            vars = root.findall(name)
            if len(vars) == 0:
                raise ValueError("Can not find %s in %s." % (name, root.tag))
            if length > 0 and len(vars) != length:
                raise ValueError(
                    "The size of %s is supposed to be %d, but is %d."
                    % (name, length, len(vars))
                )
            if length == 1:
                vars = vars[0]
            return vars

        def convert(xml_files, json_file, voc_images, sensor):
            num_obj = {'bike': 0, 'car': 0, 'car_stop': 0, 'color_cone': 0, 'person': 0}
            json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
            if self.PRE_DEFINE_CATEGORIES is not None:
                categories = self.PRE_DEFINE_CATEGORIES
            else:
                categories = get_categories(xml_files)
            bnd_id = self.START_BOUNDING_BOX_ID
            for xml_file in xml_files:
                filename, image_not_exit = get_filename(xml_file, voc_images)
                if image_not_exit:
                    continue

                try:
                    tree = ET.parse(xml_file)
                except:
                    continue
                root = tree.getroot()

                ## The filename must be a number
                image_id = get_filename_as_int(filename)
                width = self.image_size[sensor][0]
                height = self.image_size[sensor][1]
                image = {
                    "file_name": filename,
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
                json_dict["images"].append(image)
                ## Currently we do not support segmentation.
                #  segmented = get_and_check(root, 'segmented', 1).text
                #  assert segmented == '0'

                for obj in get(root, "object"):
                    category = get_and_check(obj, "name", 1).text
                    # if category=='car_stop':
                    #     category = 'car'
                    if category not in categories:
                        # new_id = len(categories)
                        # categories[category] = new_id
                        # category = 'color_cone'
                        continue
                    num_obj[category] += 1
                    category_id = categories[category]
                    bndbox = get_and_check(obj, "bndbox", 1)
                    xmin = float(get_and_check(bndbox, "xmin", 1).text)
                    ymin = float(get_and_check(bndbox, "ymin", 1).text)
                    xmax = float(get_and_check(bndbox, "xmax", 1).text)
                    ymax = float(get_and_check(bndbox, "ymax", 1).text)

                    if xmax < xmin or ymax < ymin:
                        continue
                    assert xmax > xmin
                    assert ymax > ymin

                    o_width = abs(xmax - xmin)
                    o_height = abs(ymax - ymin)
                    ann = {
                        "area": o_width * o_height,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [xmin, ymin, o_width, o_height],
                        "category_id": category_id,
                        "id": bnd_id,
                        "ignore": 0,
                        "segmentation": [],
                    }
                    json_dict["annotations"].append(ann)
                    bnd_id = bnd_id + 1
            print(num_obj)

            for cate, cid in categories.items():
                cat = {"supercategory": "none", "id": cid, "name": cate}
                json_dict["categories"].append(cat)

            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            json_fp = open(json_file, "w")
            json_str = json.dumps(json_dict)
            json_fp.write(json_str)
            json_fp.close()

        for use_type in ['train', 'test']:
            for sensor in self.sensor_type:
                xml_files = glob.glob(os.path.join(self.dataset_coco_path['rgb']['xml/xml_'+use_type], "*.xml"))
                json_file_name = os.path.join(self.dataset_coco_path[sensor]['annotations'], use_type+'.json')
                convert(xml_files, json_file_name, os.path.join(self.dataset_path[self.target_img], sensor), sensor)
        print('Finish get json ann!')

    def create_coco_format_dataset(self):
        # self.create_coco_dir()
        # self.get_image()
        # self.get_xml()
        self.create_json()

    def cal_mean_std(self):

        def image_resize(image_path, new_path, img_size=512):  # 统一图片尺寸
            for img_name in os.listdir(image_path):
                img_path = image_path + "/" + img_name  # 获取该图片全称
                image = Image.open(img_path)  # 打开特定一张图片
                image = image.resize((img_size, img_size))  # 设置需要转换的图片大小
                # process the 1 channel image
                image.save(new_path + '/' + img_name)

        def cal_mean_std(filepath, sensor, img_size=512):
            pathDir = os.listdir(filepath)

            R_channel = 0
            G_channel = 0
            B_channel = 0
            for idx in range(len(pathDir)):
                filename = pathDir[idx]
                img = imread(os.path.join(filepath, filename)) / 255.0
                R_channel = R_channel + np.sum(img[:, :, 0])
                G_channel = G_channel + np.sum(img[:, :, 1])
                B_channel = B_channel + np.sum(img[:, :, 2])

            num = len(pathDir) * img_size * img_size  # 这里（img_size, img_size）是每幅图片的大小，所有图片尺寸都一样
            R_mean = R_channel / num
            G_mean = G_channel / num
            B_mean = B_channel / num

            R_channel = 0
            G_channel = 0
            B_channel = 0
            for idx in range(len(pathDir)):
                filename = pathDir[idx]
                img = imread(os.path.join(filepath, filename)) / 255.0
                R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
                G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
                B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

            R_var = np.sqrt(R_channel / num)
            G_var = np.sqrt(G_channel / num)
            B_var = np.sqrt(B_channel / num)
            print("%s_mean = [%f, %f, %f]" % (sensor, R_mean, G_mean, B_mean))
            print("%s_std = [%f, %f, %f]" % (sensor, R_var, G_var, B_var))

        coco_main_path = os.path.join(self.ir_det_dataset_path, 'ir_det_dataset_COCO')
        if os.path.exists(os.path.join(coco_main_path, 'resize')):
            shutil.rmtree(os.path.join(coco_main_path, 'resize'))
        for sensor in self.sensor_type:
            ori_path = os.path.join(coco_main_path, sensor, 'images')  # 输入图片的文件夹路径
            new_path = os.path.join(coco_main_path, 'resize', sensor)  # resize之后的文件夹路径
            self.mkdir(new_path)
            print("# %s" % sensor)
            image_resize(ori_path, new_path)
            cal_mean_std(new_path, sensor)

if __name__ == '__main__':
    ir_det_dataset_path = '/media/vincent/856c2c04-3976-4948-ba47-5539ecaa24be/vincent/Data/ir_det_dataset'
    dt = dataset_tool(ir_det_dataset_path)
    # create coco json format dataset and output the number of different categories
    # dt.create_coco_format_dataset()
    # calculate mean and std of dataset
    dt.cal_mean_std()