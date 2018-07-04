import os
import random
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from butirecorder import Recorder
from trainers import TRAINERS
from models import MODELS



def load_data(base_dp, novel_dp, shot=5) :
    # base_train loading

    base_train = np.zeros((80, 500, 32, 32, 3))
    for label_idx, dir_name in enumerate(sorted(os.listdir(base_dp))) :
        train_fp = os.path.join(base_dp, dir_name, "train")
        for i, img_fn in enumerate(sorted(os.listdir(train_fp))) :
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp)
            img = (img - 0.5) * 2
            #img = img * 225.
            base_train[label_idx][i] = img
    """
    base_train = np.load("./base_train.npy")
    base_train = (base_train - 0.5) * 2
    """
    # base_test loading

    base_test = np.zeros((80, 100, 32, 32, 3))
    for label_idx, dir_name in enumerate(sorted(os.listdir(base_dp))) :
        test_fp = os.path.join(base_dp, dir_name, "test")
        for i, img_fn in enumerate(sorted(os.listdir(test_fp))) :
            img_fp = os.path.join(test_fp, img_fn)
            img = plt.imread(img_fp)
            img = (img - 0.5) * 2
            #img = img * 225.

            base_test[label_idx][i] = img
    """
    base_test = np.load("./base_valid.npy")
    base_test = (base_test - 0.5) * 2
    """
    # novel loading
    # img shape = (32, 32, 3), pixel range=(0, 1)
    novel_support = np.zeros((20, shot, 32, 32, 3))
    novel_test = np.zeros((20, 500 - shot, 32, 32, 3))
    for label_idx, dir_name in enumerate(sorted(os.listdir(novel_dp))) :
        train_fp = os.path.join(novel_dp, dir_name, "train")
        fn_list = os.listdir(train_fp)
        random.shuffle(fn_list)
        for i, img_fn in enumerate(fn_list) :
            img_fp = os.path.join(train_fp, img_fn)
            img = plt.imread(img_fp)
            img = (img - 0.5) * 2
            #img = img * 225.

            if i < shot:
                novel_support[label_idx][i] = img

            else:
                novel_test[label_idx][i - shot] = img

    print(base_train.shape, novel_support.shape, novel_test.shape)

    return base_train, base_test, novel_support, novel_test



def load_recorder(net_name, Model, model_version, model_index, trainer_version, record_dp, json_fn, init) :
    model = Model()
    if init :
        model.init_weight()
    recorder_name = "{}_{}".format(net_name, model_index)

    if json_fn == None :
        recorder = Recorder(
            mode="torch",
            save_mode="state_dict",
            recorder_name=recorder_name,
            save_path=record_dp,
            models={
                net_name: model,
            },
            desp="model_version: {} / trainer_version: {}".format(model_version, trainer_version),
        )

    else :
        recorder = Recorder(
            mode="torch",
            save_mode="state_dict",
            save_path=record_dp,
            models={
                net_name: model,
            },
            desp="model_version: {} / trainer_version: {}".format(model_version, trainer_version),
        )
        recorder.load(json_fn)

    return recorder




if __name__ == "__main__" :
    """ Parameters """
    NOVEL_DIR_FP = "./task2-dataset/novel/"
    BASE_DIR_FP = "./task2-dataset/base/"
    RECORDS_FP = "./records/"

    WAY = 5
    SHOT = 5
    LR = 0.0001
    EPOCH = 50


    """ Parser """
    parser = ArgumentParser()
    parser.add_argument("-i", action="store", type=int, required=True, help="model index")
    parser.add_argument("-l", action="store", type=int, default=None, help="limitation of data for training")
    parser.add_argument("-v", action="store", type=int, default=None, help="amount of validation data")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")
    parser.add_argument("--init", action="store_true", default=False, help="init weights of model")
    parser.add_argument("--step", type=int, default=None, help="limitation of steps for training")
    parser.add_argument("--lr", action="store", type=float, default=False, help="learning rate")
    parser.add_argument("--bs", action="store", type=int, default=None, help="batch size")
    parser.add_argument("--way", action="store", type=int, default=None, help="number of way")
    parser.add_argument("--shot", action="store", type=int, default=5, help="number of shot")
    parser.add_argument("--load", action="store", type=str, default=None, help="the fn of json you want to load")
    parser.add_argument("--record", action="store", type=str, required=True, help="dir path of record")
    parser.add_argument("--net", action="store", type=str, required=True, help="name of model")
    parser.add_argument("--version", action="store", type=int, default=0, help="version of model")
    parser.add_argument("--trainer", action="store", type=int, default=1, help="version of trainer")
    parser.add_argument("--seed", action="store", type=int, default=0, help="seed")

    limit = parser.parse_args().l
    valid_limit = parser.parse_args().v
    model_index = parser.parse_args().i
    cpu = parser.parse_args().cpu
    init = parser.parse_args().init
    step = parser.parse_args().step
    net_name = parser.parse_args().net
    model_version = parser.parse_args().version
    trainer_version = parser.parse_args().trainer
    seed = parser.parse_args().seed
    record_dp = parser.parse_args().record
    json_fn = parser.parse_args().load
    LR = parser.parse_args().lr if parser.parse_args().lr else LR
    WAY = parser.parse_args().way if parser.parse_args().way else WAY
    SHOT = parser.parse_args().shot if parser.parse_args().shot else SHOT

    random.seed(seed)

    """ Main """
    base_train, base_test, novel_support, novel_test = load_data(BASE_DIR_FP, NOVEL_DIR_FP, SHOT)
    recorder = load_recorder(net_name, MODELS[model_version - 1], model_version, model_index, trainer_version, record_dp, json_fn, init)
    Trainer = TRAINERS[trainer_version - 1]
    trainer = Trainer(
        recorder=recorder,
        base_train=base_train,
        base_test=base_test,
        novel_support=novel_support,
        novel_test=novel_test,
        way=WAY,
        shot=SHOT,
        cpu=cpu,
        lr=LR,
        step=step,
    )

    trainer.train()


