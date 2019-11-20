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
        print(path+' ----- folder created')
        return True
    else:
        print(path+' ----- folder existed')
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

def get_img_id():
    merged_xml_path = '/home/vincent/Data/ir_det_dataset/Annotations_ConvertedSummarized'
    night_id = extract_night(merged_xml_path)
    print('There are %s night images.' % (len(night_id)))
    merged_ok_path = '/home/vincent/Data/ir_det_dataset/vis_ok'
    merged_night_id = extract_merged_from_night(merged_ok_path, night_id)
    print('There are %s night merged images.' % (len(merged_night_id)))
    images_path = '/home/vincent/Data/ir_det_dataset/ConvertedImages'
    merged_night_id = filter_no_images(images_path, merged_night_id)
    print('There are %s night merged images after checking images existence.' % (len(merged_night_id)))
    return merged_night_id

