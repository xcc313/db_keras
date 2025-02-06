import os
import cv2
import shutil
import random
from pathlib import Path

groupdir = "E:/dm/samples/verify_code_sample/group"
pathdir = "E:/dm/samples/verify_code_sample/Train_src-2022-04/Train_src"


def group_captcha():
    files = os.listdir(pathdir)
    for f in files:
        file_path = os.path.join(pathdir, f)
        imgvalue = cv2.imread(file_path)
        group_name = '{}x{}'.format(imgvalue.shape[1], imgvalue.shape[0])
        directory = os.path.join(groupdir, group_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 跳过已经存在的,样本需要手工矫正
        if os.path.exists(os.path.join(directory, f)):
            print("{} wasdup ignore!".format(file_path))
            continue
        shutil.copyfile(file_path, os.path.join(directory, f))

lable_path = "E:/dm/samples/verify_code_sample/db_ctc"
def gen_lable(groupdir,lable_path):
    dirs = os.listdir(groupdir)
    test_percent = 0.25
    train_lines = []
    test_lines = []
    try:
        with open(os.path.join(lable_path, "train.txt"), "w", encoding='utf8') as trw,\
             open(os.path.join(lable_path, "test.txt"), "w", encoding='utf8') as tew:
            for dir in dirs:
                current_group_lines = []
                for f in os.listdir(os.path.join(groupdir, dir)):
                    file = Path(os.path.join(dir, f)).as_posix()
                    name = os.path.splitext(f)[0]
                    line = "{}\t{}".format(file, name) + '\n'
                    current_group_lines.append(line)
                random.shuffle(current_group_lines)
                image_num = len(current_group_lines)
                test_num = int(image_num * test_percent)
                test_lines += current_group_lines[:test_num]
                train_lines += current_group_lines[test_num:]
            trw.writelines(train_lines)
            tew.writelines(test_lines)
    except UnicodeEncodeError as e:
        print(e)

def gen_infer_lable(dir):
    try:
        with open(os.path.join(dir, "infer.txt"), "w", encoding='utf8') as inferw:
            current_group_lines = []
            for f in os.listdir(dir):
                if f.endswith('txt'):
                    continue
                file = Path(f).as_posix()
                name = os.path.splitext(f)[0]
                line = "{}\t{}".format(file, name) + '\n'
                current_group_lines.append(line)
            random.shuffle(current_group_lines)
            inferw.writelines(current_group_lines)
    except UnicodeEncodeError as e:
        print(e)

if __name__ == "__main__":
    gen_lable(groupdir,lable_path)
    # gen_infer_lable(r"E:/dm/samples/verify_code_sample/Train_validation")

