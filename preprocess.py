import os
import sys
import json
import random
from utils import calculate_cmvn, convert_to

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05

raw = 'raw'

label_scp_dir = raw + '/prepared_label/label_scp/'
param_scp_dir = raw + '/prepared_cmp/param_scp/'
lst_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def get_random_scp():
    label_scp = open(label_scp_dir + 'all.scp')
    param_scp = open(param_scp_dir + 'all.scp')

    label_train = open(label_scp_dir + 'train.scp', 'w')
    label_valid = open(label_scp_dir + 'valid.scp', 'w')
    label_test = open(label_scp_dir + 'test.scp', 'w')
    param_train = open(param_scp_dir + 'train.scp', 'w')
    param_valid = open(param_scp_dir + 'valid.scp', 'w')
    param_test = open(param_scp_dir + 'test.scp', 'w')

    lst_train = open(lst_dir + 'train.lst', 'w')
    lst_valid = open(lst_dir + 'valid.lst', 'w')
    lst_test = open(lst_dir + 'test.lst', 'w')

    lists_label = label_scp.readlines()
    lists_param = param_scp.readlines()

    if len(lists_label) != len(lists_param):
        print("scp files have unequal lengths")
        sys.exit(1)

    lists = list(range(len(lists_label)))
    random.seed(0)
    random.shuffle(lists)

    train_num = int(train_ratio * len(lists))
    valid_num = int(valid_ratio * len(lists))
    test_num = int(test_ratio * len(lists))
    train_lists = sorted(lists[: train_num])
    valid_lists = sorted(lists[train_num: (train_num + valid_num)])
    test_lists = sorted(lists[(train_num + valid_num):])

    for i in range(len(lists)):
        line_label = lists_label[i]
        line_param = lists_param[i]
        line_lst = line_label.strip() + ' ' + line_param.split()[1] + '\n'
        if i in valid_lists:
            label_valid.write(line_label)
            param_valid.write(line_param)
            lst_valid.write(line_lst)
        elif i in test_lists:
            label_test.write(line_label)
            param_test.write(line_param)
            lst_test.write(line_label)
        else:
            label_train.write(line_label)
            param_train.write(line_param)
            lst_train.write(line_lst)

def create_scp():
    label_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw', 'prepared_label')
    cmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw', 'prepared_cmp')

    label_files = os.listdir(label_dir)
    cmp_files = os.listdir(cmp_dir)

    label_scp = os.mkdir(os.path.join(label_dir, 'label_scp'))
    param_scp = os.mkdir(os.path.join(cmp_dir, 'param_scp'))

    label_all_scp = open(os.path.join(label_scp, 'all.scp'), 'w')
    param_all_scp = open(os.path.join(param_scp, 'all.scp'), 'w')

    for label_filename in label_files:
        label_file_path = os.path.join(label_dir, label_filename)
        label_all_scp.write(label_filename + " " + label_file_path + '\n')

    for cmp_filename in cmp_files:
        cmp_file_path = os.path.join(cmp_dir, cmp_filename)
        param_all_scp.write(cmp_filename + " " + cmp_file_path + "\n")

def main():
    create_scp()
    get_random_scp()
    #cal cmvn to the data
    calculate_cmvn('train', lst_dir, data_dir)
    convert_to('train', os.path.join(data_dir, 'train'))
    convert_to('valid', os.path.join(data_dir, 'valid'))
    convert_to('test', os.path.join(data_dir, 'test'))

if __name__ == '__main__':
    main()