import zipfile
import tqdm
from defaults import get_cfg_defaults
import sys
import logging
from net import *
import numpy as np
import random
import argparse
import os
import tensorflow as tf
import imageio
from PIL import Image

import glob
import re
import pathlib

def prepare_STL10(cfg, logger, train=True):
    directory = "./data/datasets/STL10v0/tfrecords/"
    directory2 = "./data/datasets/STL10-testv0/tfrecords/"
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory2, exist_ok=True)

    if train:
        path = "./Dataset/STL10"
        files = os.listdir(path)
        names = [f for f in files if os.path.isfile(os.path.join(path, f))]
    else:
        files = glob.glob('./Dataset/STL10_test/*/*')
        names = [f for f in files if os.path.isfile(f)]

    count = len(names)
    print("Count: %d" % count)

    random.seed(0)
    random.shuffle(names)

    folds = cfg.DATASET.PART_COUNT
    STL10_folds = [[] for _ in range(folds)]

    count_per_fold = count // folds
    for i in range(folds):
        STL10_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]

    for i in range(folds):
        images = []
        for x in tqdm.tqdm(STL10_folds[i]):
            if train:
                image_data = imageio.imread(path+"/"+x)
            else:
                image_data = imageio.imread(x)
            image = np.array(Image.fromarray(image_data).resize([128, 128]))
            images.append(image.transpose((2, 0, 1)))

        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
 
        if train:
            part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        else:
            part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        
        tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)

        random.shuffle(images)

        for image in images:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()

        for j in range(5):
            images_down = []

            for image in tqdm.tqdm(images):
                h = image.shape[1]
                w = image.shape[2]
                image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 3, h, w)
                image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)
                image_down = image_down.view(3, h // 2, w // 2).numpy()
                images_down.append(image_down)

            if train:
                part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)
            else:
                part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)
            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            
            for image in images_down:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
                tfr_writer.write(ex.SerializeToString())
            tfr_writer.close()

            images = images_down


def run():
    parser = argparse.ArgumentParser(description="ALAE. Prepare tfrecords for celeba128x128")
    parser.add_argument(
        "--config-file",
        default="configs/STL10.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prepare_STL10(cfg, logger, True)
    print("train-finish")
    prepare_STL10(cfg, logger, False)


if __name__ == '__main__':
    run()

