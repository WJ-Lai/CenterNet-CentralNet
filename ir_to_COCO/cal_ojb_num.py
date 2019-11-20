import os
import random
import shutil
import sys
import json
import glob
import xml.etree.ElementTree as ET
import re

#from https://www.php.cn/python-tutorials-424348.html
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        # print(path+' ----- folder created')
        return True
    else:
        # print(path+' ----- folder existed')
        return False
#foler to make, please enter full path


"""
main code below are from
https://github.com/Tony607/voc2coco
"""


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


def convert(xml_files, json_file, voc_images):
    num_obj = {'bike': 0, 'car': 0, 'car_stop': 0, 'color_cone': 0, 'person': 0}
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        filename, image_not_exit = get_filename(xml_file, voc_images)
        if image_not_exit:
            continue

        try:
            tree = ET.parse(xml_file)
        except:
            continue
        root = tree.getroot()
        # root = get_and_check(root, sensor, 1)
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        width = image_size[sensor][0]
        height = image_size[sensor][1]
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


def create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    else:
        mkdir(dir_name)


def get_fileID(xml_file):
    filename = xml_file.split('/')[-1]
    filename = filename.split('.')[0]
    return filename


def extract_night(merge_xml_path):
    xml_files = os.listdir(merge_xml_path)
    night_id = []
    for xml_file in xml_files:
        try:
            tree = ET.parse(os.path.join(merge_xml_path, xml_file))
        except:
            continue
        root = tree.getroot()
        time = get_and_check(root, "time", 1).text
        if time == 'night':
                night_id.append(get_fileID(xml_file))
    return night_id


def extract_merged_from_night(merged_ok_path, night_id):
    merge_ok_files = os.listdir(merged_ok_path)
    merged_night_id = []
    for merge_ok_file in merge_ok_files:
        fileID = get_fileID(merge_ok_file)
        if fileID in night_id:
            merged_night_id.append(fileID)
    return merged_night_id


def filter_no_images(images_path, merged_night_id):
    sensor_types = ['rgb', 'fir', 'mir', 'nir']
    merged_night_id = set(merged_night_id)
    for sensor in sensor_types:
        sensor_path = os.path.join(images_path, sensor)
        sensor_imges = os.listdir(os.path.join(images_path, sensor))
        for idx, sensor_imge in enumerate(sensor_imges):
            sensor_imges[idx] = get_fileID(sensor_imge)
        sensor_imges = set(sensor_imges)
        merged_night_id = merged_night_id & sensor_imges
    return merged_night_id


"""
You only need to set the following three parts
1.val_files_num : num of validation samples from your all samples
2.test_files_num = num of test samples from your all samples
3.voc_annotations : path to your VOC dataset Annotations

"""


# get merged and check file existence
from ir_to_COCO.extract_night_merge_well import get_img_id
merged_night_id = get_img_id()

val_num_rate = 0.2
test_num_rate = 0.2
sensor_type = ['rgb', 'fir', 'mir', 'nir']
# image_size = {'rgb':[640, 480], 'fir':[640, 480], 'mir':[320, 256], 'nir':[320, 256]}
# if ConvertedImages
image_size = {'rgb': [320, 256], 'fir': [320, 256], 'mir': [320, 256], 'nir': [320, 256]}

START_BOUNDING_BOX_ID = 0
PRE_DEFINE_CATEGORIES = None
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {"bike": 0, "car": 1, 'car_stop': 2, "color_cone": 3, "person": 4}
PRE_DEFINE_CATEGORIES = {"bike": 0, "car": 1, "color_cone": 2, "person": 3}

val_files_num = int(len(merged_night_id)*val_num_rate)
test_files_num = int(len(merged_night_id)*test_num_rate)
# voc_annotations = '/home/vincent/Data/ir_det_dataset/Annotations_Converted/'  # remember to modify the path
voc_annotations = '/home/vincent/Data/ir_det_dataset/Annotations_ConvertedSummarized/'  # remember to modify the path

split = voc_annotations.split('/')
coco_name = split[-3]
del split[-2]
del split[-1]
del split[0]
# print(split)
main_path = ''
for i in split:
    main_path += '/' + i

main_path = main_path + '/'


for sensor in sensor_type:
    print(sensor)
    coco_path = os.path.join(main_path, coco_name + '_COCO/', sensor)
    coco_images = os.path.join(main_path, coco_name + '_COCO', sensor, 'images/')
    coco_json_annotations = os.path.join(main_path, coco_name + '_COCO', sensor, 'annotations/')
    # xml_val = os.path.join(main_path, coco_name + '_COCO/', sensor, 'xml', 'xml_val/')
    # xml_test = os.path.join(main_path, coco_name + '_COCO/', sensor, 'xml/', 'xml_test/')
    xml_train = os.path.join(main_path, coco_name + '_COCO/', sensor, 'xml/', 'xml_train/')

    voc_images = os.path.join(main_path, 'ConvertedImages', sensor)

    dir_path = [coco_path, coco_images, coco_json_annotations,  xml_train]
    for dir_name in dir_path:
        create_dir(dir_name)

    # voc annotations copy to coco annotations
    for i in os.listdir(voc_annotations):
        id = get_fileID(i)
        if id in merged_night_id:
            img_path = os.path.join(voc_annotations, i)
            shutil.copy(img_path, xml_train)


    # create json
    xml_train_files = glob.glob(os.path.join(xml_train, "*.xml"))

    convert(xml_train_files, coco_json_annotations + 'train.json', voc_images)
