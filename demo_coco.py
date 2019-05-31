#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019
#
# Date:   30 May 2019
# COCO dataset has 182 classes and person belongs to class 0 in the dataset.
# Segmented only the person class from the image.

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
import os
import shutil

from libs.models import *
from libs.utils import DenseCRF

OUTPUT_DIR = './outpt/'
FRAME_SUBFOLDER = 'frame/'

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


def get_frame(vidcap, sec):
    # vidcap.set(cv2.CAP_PROS_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    return hasFrames, image


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-v",
    "--video-path",
    type=click.Path(exists=True),
    required=True,
    help="Video to be processed",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
def single(config_path, model_path, video_path, cuda, crf):
    """
    Segment person from a video
    """

    # Setup
    CONFIG = Dict(yaml.load(config_path))
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Read frames from video.
    vidcap = cv2.VideoCapture(video_path)
    # vidcap.set(cv2.CAP_PROS_FPS, 20)
    success, image = vidcap.read()
    sec = 0
    frameRate = 0.3
    count = 0
    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        # Inference
        image, raw_image = preprocessing(image, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)      
        if 0 not in np.unique(labelmap):
            print("Person not found in the frame.")
            success, image = get_frame(vidcap, sec)
        else:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            if not os.path.exists(OUTPUT_DIR+FRAME_SUBFOLDER):
                os.makedirs(OUTPUT_DIR+FRAME_SUBFOLDER)
            mask = labelmap == 0
            image = raw_image.copy()
            image[mask==False] = 255
            plt.figure()
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(OUTPUT_DIR+FRAME_SUBFOLDER+'output'+str(count)+'.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
            count+=1
            print("Frame ", count, " completed.")
            success, image = get_frame(vidcap, sec)
            if count == 5:
                break
    vidcap.release()

    # Write the stacked image to the output directory.
    if os.path.exists(OUTPUT_DIR+FRAME_SUBFOLDER): # if the folder exists, at least one image is present in the directory.
        image = cv2.imread(OUTPUT_DIR+FRAME_SUBFOLDER+'output0.png')
        print(image.shape)
        image = cv2.resize(image, (64, 64))
        for i in range(1, count):
            img = cv2.imread(OUTPUT_DIR+FRAME_SUBFOLDER+'output'+str(i)+'.png')
            img = cv2.resize(img, (64, 64))
            image = np.concatenate((image, img), axis = 1)
        cv2.imwrite(OUTPUT_DIR+'stacked_output.png', image)
        
        # remove the subfolder.
        # os.rmdir(OUTPUT_DIR+FRAME_SUBFOLDER)
        shutil.rmtree(OUTPUT_DIR+FRAME_SUBFOLDER)

if __name__ == "__main__":
    main()

