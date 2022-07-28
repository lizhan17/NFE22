from __future__ import division
from __future__ import print_function

import torch
import os
import time
import numpy as np
import pickle
import cv2

# class blur(object):
#     def __init__(self, size):
#
#     def __call__(self, img):
#         return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]

def flip(img):
    return np.fliplr(img)

def group_random_flip(img_group):
    return [flip(img) for img in img_group]

def rotation(img, degree, interpolation=cv2.INTER_LINEAR, value=0):
    h, w = img.shape[0:2]
    center = (w / 2, h / 2)
    map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    img = cv2.warpAffine(
        img,
        map_matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value)
    return img


def group_rotation(img_group, degree, interpolations, values):
    outs = list()
    for img, interpolation, value in zip(img_group, interpolations, values):
        outs.append(rotation(img, degree, interpolation, value))
    return outs


class StaticResize(object):

    def __init__(self):
        pass
    def __call__(self, img):
        dim = (256, 256)
        # resize image
        h, w, c = img.shape
        if c == 3:
            #img = img.astype('float32') we use uint8 after resize
            resized = cv2.resize(img, dsize=dim, interpolation= cv2.INTER_LINEAR)

        elif c == 2:
            flow1 = img[:,:,0]
            flow2 = img[:,:,1]
            newflow = cv2.merge((flow1, flow2, flow1))
            resized = cv2.resize(newflow, dsize=dim, interpolation=cv2.INTER_LINEAR)
            resized = resized[:, :,:2]

        elif c == 1:
            img = img.astype('uint8')
            newmask = cv2.merge((img, img, img))
            resized = cv2.resize(newmask, dsize=dim, interpolation=cv2.INTER_LINEAR)
            resized = resized[:, :, 0:1]


        return resized


class StaticRandomCrop(object):
    """
    Helper function for random spatial crop
    """
    def __init__(self, size, image_shape):
        h, w = image_shape

        self.th, self.tw = size
        self.h1 = torch.randint(0, h - self.th + 1, (1,)).item()
        self.w1 = torch.randint(0, w - self.tw + 1, (1,)).item()
    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]

class FixedCrop(object):
    """
    Helper function for random spatial crop
    """
    def __init__(self, size, image_shape):
        h, w = image_shape
        self.th, self.tw = size
        self.h1 = 10
        self.w1 = 10

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def getflowfromimagepath(inputimagepaths, flowfiletype):
    inputflowpaths = []
    if "tt" in flowfiletype:
        for eachpngfile in inputimagepaths:
            imagedir = os.path.dirname(eachpngfile)
            basename = os.path.basename(eachpngfile)

            newbasename = basename.replace("_left.png","")
            baseid = int(newbasename)
            nextbasename = str(baseid + 1).zfill(len(newbasename))
            flowname = newbasename + "_" + nextbasename + "_flow.npy"
            flowdir = imagedir.replace("image_left","flow")
            flowpath = flowdir + "/" + flowname

            assert os.path.exists(flowpath)
            flowdata = np.load(flowpath, allow_pickle=True)

            inputflowpaths.append(flowdata)
    if "sintel" in flowfiletype:
        for eachpng in inputimagepaths:
            basename = os.path.basename(eachpng)
            assert basename.startswith(("frame_"))
            flowpath = eachpng.replace(".png",".flo").replace("/final","/flow")
            assert os.path.exists(flowpath)
            flowdata = readFlowFile(flowpath)
            inputflowpaths.append(flowdata)
    if "ours" in flowfiletype:
        for eachpng in inputimagepaths:
            flowpath = eachpng.replace(".png", ".npy").replace("/final", "/flow")
            assert os.path.exists(flowpath)
            flowdata = np.load(flowpath, allow_pickle=True)
            inputflowpaths.append(flowdata)


    return inputflowpaths