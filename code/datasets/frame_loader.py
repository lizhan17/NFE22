from __future__ import division
from __future__ import print_function

import os
import natsort
import numpy as np
import cv2
import glob

import torch
from torch.utils import data
from datasets.dataset_utils import StaticRandomCrop
from datasets.dataset_utils import FixedCrop
from datasets.dataset_utils import getflowfromimagepath
from datasets.dataset_utils import load_pkl
from datasets.dataset_utils import group_rotation, rotation, group_random_flip, flip

import random

import time
import imageio

class FrameLoader_ours_cropped_sqtrain(data.Dataset):
    def __init__(self, args, root, is_training = False, transform=None):
        print("creating train ###########################################", root)
        print("is", is_training)


        self.is_training = is_training
        self.transform = transform
        self.chsize = 3
        self.inputmasklevel = args.inputmasklevel


        self.inputframetype  = 0
      
 
        self.inputmasktype = 1
    
        # carry over command line arguments
        self.sequence_length = args.sequence_length + args.futurestep - 1  # 2>3  should be input 4-1
        sequence_length = self.sequence_length + 1
        assert args.sequence_length > 1, 'sequence length must be > 1'

        assert args.sample_rate > 0, 'sample rate must be > 0'
        self.sample_rate = args.sample_rate

        self.crop_size = args.crop_size
        self.start_index = args.start_index
        self.stride = args.stride

        assert (os.path.exists(root))
        if self.is_training:
            self.start_index = 0


        
        dirs = glob.glob(root + "/p*/s*/final_pkl/")  #
        dirs = natsort.natsorted(dirs)
        datasets = []
        # create sequences
        for eachdir in dirs:
            pngs = os.listdir(eachdir)
            pngs = [f for f in pngs if f.endswith(".pkl")]
            pngs = natsort.natsorted(pngs)
            startindx = int(pngs[0].replace(".pkl", ""))
            length = len(pngs)

            numseq = int (length / sequence_length)       # comment
            for seqidx in range(numseq):                  # comment
                i = seqidx * sequence_length + startindx  # comment
                sequence = []
                for j in range(0, sequence_length):
                    indexid = i + j    #

                    pngpath = os.path.join(eachdir, str(indexid) + ".pkl")
                    assert (os.path.exists(pngpath))
                    sequence.append(pngpath)
                datasets.append(sequence)
        print("fully sequence", len(datasets))
        # create dict
        # datasets = datasets[0:2]

        self.datasetdict = {}



        for index in range(0, len(datasets)):
            self.datasetdict[index] = {}
            pkpaths = datasets[index]

            self.datasetdict[index]["image"] = pkpaths
            for f in pkpaths:
                assert (os.path.exists(f))

        
            self.datasetdict[index]["mask"] = [f.replace("/final_pkl", "/finalmask0" + str(self.inputmasklevel) + "_pkl") for f in pkpaths[1:]]
            for f in self.datasetdict[index]["mask"]:
                assert (os.path.exists(f))
        

            self.datasetdict[index]["gt_flow"] = [f.replace("/final_pkl", "/flow_pkl") for f in pkpaths[1:-1]]



    def __len__(self):
        return len(self.datasetdict.keys())

    def __getitem__(self, index):

        pathdict = self.datasetdict[index]
        output_dict = {}
        # flip = random.random() < 0.5

        if "mask" in pathdict:
            output_dict["mask"] = [load_pkl(f) for f in pathdict["mask"]]

        if "gt_flow" in pathdict:
            output_dict["gt_flow"] = [load_pkl(f) for f in pathdict["gt_flow"]]

        output_dict["image"] = [load_pkl(f) for f in pathdict["image"]]


        input_shape = output_dict["image"][0].shape[:2]
        cropper = StaticRandomCrop(self.crop_size, input_shape)
        for k in output_dict:
            tmp = map(cropper, output_dict[k])
            output_dict[k] = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in tmp]
        return output_dict


class FrameLoader_ours_cropped_sqtest(data.Dataset):
    def __init__(self, args, root, is_training = False, transform=None):
        self.is_training = is_training
        self.transform = transform
        self.chsize = 3
        self.inputmasklevel = args.inputmasklevel


        ## for input frames
        self.inputframetype  = 0
       

        # carry over command line arguments
        assert args.sequence_length > 1, 'sequence length must be > 1'
        self.sequence_length = args.sequence_length + args.futurestep   # 2>3  should be input 4-1

        assert args.sample_rate > 0, 'sample rate must be > 0'
        self.sample_rate = args.sample_rate

        self.crop_size = args.crop_size
        self.start_index = args.start_index
        self.stride = args.stride

        assert (os.path.exists(root))
        if self.is_training:
            self.start_index = 0

        # collect, colors, motion vectors, and depth


        sequence_length = self.sequence_length 
        dirs = glob.glob(root + "/p*/s*/final/")  #
        dirs = natsort.natsorted(dirs)
        print(dirs)
        datasets = []
        # create sequences
        for eachdir in dirs:
            pngs = os.listdir(eachdir)
            pngs = [f for f in pngs if f.endswith(".png")]
            pngs = natsort.natsorted(pngs)
            startindx = int(pngs[0].replace(".png", ""))
            length = len(pngs)

            numseq = int (length / sequence_length)       # comment
            for seqidx in range(numseq):                  # comment
                i = seqidx * sequence_length + startindx  # comment
                sequence = []
                for j in range(0, sequence_length):
                    indexid = i + j    #

                    pngpath = os.path.join(eachdir, str(indexid) + ".png")
                    assert (os.path.exists(pngpath))
                    sequence.append(pngpath)
                datasets.append(sequence)
        print("fully sequence", len(datasets))

   

        self.datasetdict = {}

        for index in range(0, len(datasets)):
            self.datasetdict[index] = {}
            pkpaths = datasets[index]

            self.datasetdict[index]["image"] = pkpaths
            for f in pkpaths:
                assert (os.path.exists(f))


            self.datasetdict[index]["mask"] = [f.replace("/final", "/finalmask0" + str(self.inputmasklevel)).replace(".png", ".npy") for f in pkpaths[1:]]
            for f in self.datasetdict[index]["mask"]:
                # assert (os.path.exists(f))
                if not os.path.exists(f):
                    print("missing: ", f)
                    quit()
  

            self.datasetdict[index]["gt_flow"] = [f.replace("/final", "/flow").replace(".png", ".npy") for f in pkpaths[1:-1]]

    def __len__(self):
        return len(self.datasetdict.keys())

    def __getitem__(self, index):

        pathdict = self.datasetdict[index]
        output_dict = {}
        if "mask" in pathdict:
            output_dict["mask"] = [np.load(f, allow_pickle=True) for f in pathdict["mask"]]
            output_dict["mask"] = [torch.from_numpy(im[..., np.newaxis].transpose(2, 0, 1)).float() for im in output_dict["mask"]]

    
        if "gt_flow" in pathdict:
            output_dict["gt_flow"] = [np.load(f, allow_pickle=True) for f in pathdict["gt_flow"]]
            output_dict["gt_flow"] = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in output_dict["gt_flow"]]

        output_dict["image"] = [cv2.imread(f) for f in pathdict["image"]] # 
        output_dict["image"] = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in output_dict["image"]] # to channel first
        output_dict["input_files"] = pathdict["image"] #for evaluation saving images

        return output_dict

