from glob import glob
import os
import random
import pandas as pd
from config import *


# 根据标注文件生成对应关系
def get_annotation(ann_path):
    with open(ann_path) as file:
        anns = {}
        for line in file.readlines():
            arr = line.split('\t')[1].split()
            name = arr[0]
            start = int(arr[1])
            end = int(arr[-1])
            # 标注太长，可能有问题
            if end - start > 50:
                continue
            anns[start] = 'B-' + name
            for i in range(start + 1, end):
                anns[i] = 'I-' + name
        return anns


def get_text(txt_path):
    with open(txt_path) as file:
        return file.read()


# 建立文字和标签对应关系
def generate_annotation():
    for txt_path in glob(ORIGIN_DIR + '*.txt'):
        ann_path = txt_path[:-3] + 'ann'
        anns = get_annotation(ann_path)
        text = get_text(txt_path)
        # 建立文字和标注对应
        df = pd.DataFrame({'word': list(text), 'label': ['O'] * len(text)})
        df.loc[anns.keys(), 'label'] = list(anns.values())
        # 导出文件
        file_name = os.path.split(txt_path)[1]
        df.to_csv(ANNOTATION_DIR + file_name, header=None, index=None)


# 拆分训练集和测试集
def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + '*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    # 合并文件
    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)


def merge_file(files, target_path):
    with open(target_path, 'a') as file:
        for f in files:
            text = open(f).read()
            file.write(text)


if __name__ == '__main__':
    # 建立文字和标签对应关系
    # generate_annotation()

    # 拆分训练集和测试集
    split_sample()
