
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import os
import cv2
from models.model_utils import conv2d, deconv2d
from models import common
import numpy as np

#from flownet2_pytorch.models import FlowNet2
#from flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
import torch.nn.functional as F
import time

import torch.nn.functional as F
import time

backwarp_tenGrid = {}
# ref https://github.com/sniklaus/softmax-splatting/blob/e44793c99456151521e95db50432eb7403199de0/run.py#L34
def Resample2d(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)


# this code does not need flownet2 resample2d package used in our paper
# but it is very close to our reported results(<0.01 dB).
class FPCGE(nn.Module):
    def __init__(self, args):
        super(FPCGE, self).__init__()
        ## for gain adaptive

        self.threshold = args.threshold
        ##
        self.rgb_max = args.rgb_max
        self.sequence_length = args.sequence_length

        # our defined
        self.futurestep = args.futurestep
        self.plossratio = args.pr
        self.bceratio = args.br
        self.costratio = args.cr
        self.temperalratio = args.tr


        self.lossfunction = args.lossfunction


        self.inverseflow = None

        self.phase = args.phase

        self.timing = args.timing


        factor = 2

        # flow prediction from frame, flow and mask

        input_channels = (self.sequence_length - 1 ) * 2 + (self.sequence_length) * 3 + 1
        out_channels = 2

        self.conv1 = nn.ModuleList([conv2d(input_channels, 64 // factor, kernel_size=7, stride=2) for i in range(self.futurestep)])
        self.conv2 = nn.ModuleList([conv2d(64 // factor, 128 // factor, kernel_size=5, stride=2) for i in range(self.futurestep)])
        self.conv3 = nn.ModuleList([conv2d(128 // factor, 256 // factor, kernel_size=5, stride=2) for i in range(self.futurestep)])
        self.conv3_1 = nn.ModuleList([conv2d(256 // factor, 256 // factor) for i in range(self.futurestep)])
        self.conv4 = nn.ModuleList([conv2d(256 // factor, 512 // factor, stride=2) for i in range(self.futurestep)])
        self.conv4_1 = nn.ModuleList([conv2d(512 // factor, 512 // factor) for i in range(self.futurestep)])
        self.conv5 = nn.ModuleList([conv2d(512 // factor, 512 // factor, stride=2) for i in range(self.futurestep)])
        self.conv5_1 = nn.ModuleList([conv2d(512 // factor, 512 // factor) for i in range(self.futurestep)])
        self.conv6 = nn.ModuleList([conv2d(512 // factor, 1024 // factor, stride=2) for i in range(self.futurestep)])
        self.conv6_1 = nn.ModuleList([conv2d(1024 // factor, 1024 // factor) for i in range(self.futurestep)])
        self.deconv5 = nn.ModuleList([deconv2d(1024 // factor, 512 // factor) for i in range(self.futurestep)])
        self.deconv4 = nn.ModuleList([deconv2d(1024 // factor, 256 // factor) for i in range(self.futurestep)])
        self.deconv3 = nn.ModuleList([deconv2d(768 // factor, 128 // factor) for i in range(self.futurestep)])
        self.deconv2 = nn.ModuleList([deconv2d(384 // factor, 64 // factor) for i in range(self.futurestep)])
        self.deconv1 = nn.ModuleList([deconv2d(192 // factor, 32 // factor) for i in range(self.futurestep)])
        self.deconv0 = nn.ModuleList([deconv2d(96 // factor, 16 // factor) for i in range(self.futurestep)])
        self.final_flow = nn.ModuleList([nn.Conv2d(input_channels + 16 // factor, out_channels,
                                    kernel_size=3, stride=1, padding=1, bias=True) for i in range(self.futurestep)])



        input_channels_c = input_channels

        self.conv1_c = nn.ModuleList(
            [conv2d(input_channels_c, 64 // factor, kernel_size=7, stride=2) for i in range(self.futurestep)])
        self.conv2_c = nn.ModuleList(
            [conv2d(64 // factor, 128 // factor, kernel_size=5, stride=2) for i in range(self.futurestep)])
        self.conv3_c = nn.ModuleList(
            [conv2d(128 // factor, 256 // factor, kernel_size=5, stride=2) for i in range(self.futurestep)])
        self.conv3_1_c = nn.ModuleList([conv2d(256 // factor, 256 // factor) for i in range(self.futurestep)])
        self.conv4_c = nn.ModuleList([conv2d(256 // factor, 512 // factor, stride=2) for i in range(self.futurestep)])
        self.conv4_1_c = nn.ModuleList([conv2d(512 // factor, 512 // factor) for i in range(self.futurestep)])
        self.conv5_c = nn.ModuleList([conv2d(512 // factor, 512 // factor, stride=2) for i in range(self.futurestep)])
        self.conv5_1_c = nn.ModuleList([conv2d(512 // factor, 512 // factor) for i in range(self.futurestep)])
        self.conv6_c = nn.ModuleList([conv2d(512 // factor, 1024 // factor, stride=2) for i in range(self.futurestep)])
        self.conv6_1_c = nn.ModuleList([conv2d(1024 // factor, 1024 // factor) for i in range(self.futurestep)])

        self.deconv5_c = nn.ModuleList([deconv2d(1024 // factor, 512 // factor) for i in range(self.futurestep)])
        self.deconv4_c = nn.ModuleList([deconv2d(1024 // factor, 256 // factor) for i in range(self.futurestep)])
        self.deconv3_c = nn.ModuleList([deconv2d(768 // factor, 128 // factor) for i in range(self.futurestep)])
        self.deconv2_c = nn.ModuleList([deconv2d(384 // factor, 64 // factor) for i in range(self.futurestep)])
        self.deconv1_c = nn.ModuleList([deconv2d(192 // factor, 32 // factor) for i in range(self.futurestep)])
        self.deconv0_c = nn.ModuleList([deconv2d(96 // factor, 16 // factor) for i in range(self.futurestep)])
        self.final_flow_c = nn.ModuleList([
            nn.Conv2d(input_channels_c + 16 // factor, 1, kernel_size=3, stride=1, padding=1, bias=True) for i in
            range(self.futurestep)])

        input_channels_g = 3 + 3

        self.conv1_g = nn.ModuleList(
            [conv2d(input_channels_g, 64 // factor, kernel_size=7, stride=2) for i in range(self.futurestep)])
        self.conv2_g = nn.ModuleList(
            [conv2d(64 // factor, 128 // factor, kernel_size=5, stride=2) for i in range(self.futurestep)])
        self.conv3_g = nn.ModuleList(
            [conv2d(128 // factor, 256 // factor, kernel_size=5, stride=2) for i in range(self.futurestep)])
        self.conv3_1_g = nn.ModuleList([conv2d(256 // factor, 256 // factor) for i in range(self.futurestep)])
        self.conv4_g = nn.ModuleList([conv2d(256 // factor, 512 // factor, stride=2) for i in range(self.futurestep)])
        self.conv4_1_g = nn.ModuleList([conv2d(512 // factor, 512 // factor) for i in range(self.futurestep)])
        self.conv5_g = nn.ModuleList([conv2d(512 // factor, 512 // factor, stride=2) for i in range(self.futurestep)])
        self.conv5_1_g = nn.ModuleList([conv2d(512 // factor, 512 // factor) for i in range(self.futurestep)])
        self.conv6_g = nn.ModuleList([conv2d(512 // factor, 1024 // factor, stride=2) for i in range(self.futurestep)])
        self.conv6_1_g = nn.ModuleList([conv2d(1024 // factor, 1024 // factor) for i in range(self.futurestep)])

        self.deconv5_g = nn.ModuleList([deconv2d(1024 // factor, 512 // factor) for i in range(self.futurestep)])
        self.deconv4_g = nn.ModuleList([deconv2d(1024 // factor, 256 // factor) for i in range(self.futurestep)])
        self.deconv3_g = nn.ModuleList([deconv2d(768 // factor, 128 // factor) for i in range(self.futurestep)])
        self.deconv2_g = nn.ModuleList([deconv2d(384 // factor, 64 // factor) for i in range(self.futurestep)])
        self.deconv1_g = nn.ModuleList([deconv2d(192 // factor, 32 // factor) for i in range(self.futurestep)])
        self.deconv0_g = nn.ModuleList([deconv2d(96 // factor, 16 // factor) for i in range(self.futurestep)])
        self.final_gain = nn.ModuleList([
            nn.Conv2d(input_channels_g + 16 // factor, 3, kernel_size=3, stride=1, padding=1, bias=True) for i in
            range(self.futurestep)])


        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

        self.warp_bilinear = Resample2d

        self.L1Loss = nn.L1Loss()
        self.bcloss = torch.nn.BCEWithLogitsLoss()

        if self.phase == "fused":
            pass
            # common.freezeweights(
            #     [self.conv1, self.conv2, self.conv3, self.conv3_1, self.conv4, self.conv4_1, self.conv5, self.conv5_1,
            #      self.conv6, self.conv6_1,
            #      self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5, self.deconv0, self.final_flow, self.grucell])


        self.crossloss = nn.CrossEntropyLoss()

  
    def preprocess(self, input_images, input_flows, input_mask):
        concated_images = torch.cat([image.unsqueeze(2) for image in input_images], dim=2).contiguous()
        rgb_mean = concated_images.view(concated_images.size()[:2] + (-1,)).mean(dim=-1).view(
            concated_images.size()[:2] + 2 * (1,))
        input_images = [(input_image - rgb_mean) / self.rgb_max for input_image in input_images]
        bsize, channels, height, width = input_flows[0].shape

        input_images = torch.cat([input_image.unsqueeze(2) for input_image in input_images], dim=2)
        input_images = input_images.contiguous().view(bsize, -1, height, width)

        bsize, channels, height, width = input_flows[0].shape
        input_flows = torch.cat([input_flow.unsqueeze(2) for input_flow in input_flows], dim=2)
        input_flows = input_flows.contiguous().view(bsize, -1, height, width)

        if input_mask is not None:
           images_and_flows_masks = torch.cat((input_flows, input_images , input_mask), dim=1)
        else:
            images_and_flows_masks =  torch.cat((input_flows, input_images), dim=1) #torch.cat((input_flows), dim=1)
        return images_and_flows_masks



    def flow_encoder(self, images_and_flows, i):

        out_conv1 = self.conv1[i](images_and_flows)
        out_conv2 = self.conv2[i](out_conv1)
        out_conv3 = self.conv3_1[i](self.conv3[i](out_conv2))
        out_conv4 = self.conv4_1[i](self.conv4[i](out_conv3))
        out_conv5 = self.conv5_1[i](self.conv5[i](out_conv4))
        out_conv6 = self.conv6_1[i](self.conv6[i](out_conv5))  # [4, 512, 4, 4])
        features = [out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6]
        return features

    def flow_decoder(self, features, images_and_flows, i):
        out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = features
        out_deconv5 = self.deconv5[i](out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)
        out_deconv4 = self.deconv4[i](concat5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3[i](concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2[i](concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1[i](concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0[i](concat1)
        concat0 = torch.cat((images_and_flows, out_deconv0), 1)
        flows = self.final_flow[i](concat0)
        return flows

    def confident_encoder(self, images_and_flows, i):

        out_conv1 = self.conv1_c[i](images_and_flows)
        out_conv2 = self.conv2_c[i](out_conv1)
        out_conv3 = self.conv3_1_c[i](self.conv3_c[i](out_conv2))
        out_conv4 = self.conv4_1_c[i](self.conv4_c[i](out_conv3))
        out_conv5 = self.conv5_1_c[i](self.conv5_c[i](out_conv4))
        out_conv6 = self.conv6_1_c[i](self.conv6_c[i](out_conv5))  # [4, 512, 4, 4])
        features = [out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6]
        return features

    def confident_decoder(self, features, images_and_flows, i):
        out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = features

        out_deconv5 = self.deconv5_c[i](out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)
        out_deconv4 = self.deconv4_c[i](concat5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3_c[i](concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2_c[i](concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1_c[i](concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0_c[i](concat1)
        concat0 = torch.cat((images_and_flows, out_deconv0), 1)
        output_confident = self.final_flow_c[i](concat0)

        return output_confident

    def network_output_g(self, input_images, input_gains, i):

        bsize, channels, height, width = input_images.shape
        input_images = torch.cat([input_image.unsqueeze(2) for input_image in input_images], dim=2)
        input_images = input_images.contiguous().view(bsize, -1, height, width)

        images_and_gains = torch.cat((input_images, input_gains), dim=1)
        out_conv1 = self.conv1_g[i](images_and_gains)
        out_conv2 = self.conv2_g[i](out_conv1)
        out_conv3 = self.conv3_1_g[i](self.conv3_g[i](out_conv2))
        out_conv4 = self.conv4_1_g[i](self.conv4_g[i](out_conv3))
        out_conv5 = self.conv5_1_g[i](self.conv5_g[i](out_conv4))
        out_conv6 = self.conv6_1_g[i](self.conv6_g[i](out_conv5))

        out_deconv5 = self.deconv5_g[i](out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)

        out_deconv4 = self.deconv4_g[i](concat5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)

        out_deconv3 = self.deconv3_g[i](concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)

        out_deconv2 = self.deconv2_g[i](concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        out_deconv1 = self.deconv1_g[i](concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)

        out_deconv0 = self.deconv0_g[i](concat1)

        concat0 = torch.cat((images_and_gains, out_deconv0), 1)
        output = self.final_gain[i](concat0)
        return output


    def forward(self, input_dict):
        if self.timing:
            torch.cuda.synchronize()  # syncuda
            starttime = time.perf_counter()
         
        images = input_dict['image']
        images_except_target = images[:self.sequence_length]
        target_image_list = images[self.sequence_length:]

        input_flows = input_dict["gt_flow"][:self.sequence_length-1]

        losses = {}
        predframehistory = []
        flowhistory = []
        maskhistory = []
        input_masks = input_dict["mask"][:self.sequence_length - 1]

        image_prediction_fusedhistory = []

        addgainlist = []
        beforeenhancelist = []
        gtconfidentlist = []

        for futureframeoffset in range(0, self.futurestep): # [3,4,5] futurestep =3
            if futureframeoffset == 0:
                lastimage = images_except_target[-1].clone()
                last_flow_gain = (input_flows[-1]).clone()

                last2_image = (images_except_target[-2]).clone()
                assert (len(images_except_target) == 2)

                inputflowlist = [flow.clone() for flow in input_flows]
                inputflowlistprev = [flow.clone() for flow in inputflowlist]
                inputframelist = [frame.clone() for frame in images_except_target]
                inputframelistprev = [frame.clone() for frame in inputframelist]
                pastconfident = input_masks[-1].clone()
            else:
                inputflowlistprev.pop(0)
                inputflowlistprev.append(flowpredictionprev)
                inputflowlist = [f.clone() for f in inputflowlistprev]

                inputframelistprev.pop(0)
                inputframelistprev.append(imagepredictionprev)
                inputframelist = [f.clone() for f in inputframelistprev]

                lastimage = imagepredictionprev.clone()

                last_flow_gain = (inputflowlist[-1]).clone()
                assert (len(inputflowlist) == 1)
                last2_image = (inputframelist[-2]).clone()
                assert (len(inputframelist) == 2)

            # I1 I2 and flow2
            # I1 to I2
            warped_last2 = self.warp_bilinear(last2_image, last_flow_gain)
            last_gain = (lastimage - warped_last2) # I2 - warped(I1)



            imageandflows = self.preprocess(inputframelist, inputflowlist, pastconfident) #we use pastconfident
            flowfeatures = self.flow_encoder(imageandflows, futureframeoffset)
            imageandflowmask = self.preprocess(inputframelist, inputflowlist, pastconfident)
            confidentfeatures = self.confident_encoder(imageandflowmask, futureframeoffset)

            flowprediction = self.flow_decoder(flowfeatures, imageandflows, futureframeoffset) #predict flow
            flowpredictionprev = flowprediction.clone()

            if self.phase == "fused": # we also decoder confidender

                futuremask = self.confident_decoder(confidentfeatures, imageandflowmask, futureframeoffset) # predict mask

                maskcopy = futuremask.clone() # used for bce loss . bce already take sigmoid.
                futuremask = torch.sigmoid(futuremask)

                futuremask = futuremask > self.threshold #orginal is 0.4 we use

                futuremask = futuremask.float()
                #print("after", torch.mean(futuremask))

                pastconfident = futuremask.clone()

                imageprediction_withoutmask = self.warp_bilinear(lastimage, flowprediction)

            
                warped_future_gain = self.warp_bilinear(last_gain, flowprediction) # warp(gain2) to 3

                imageprediction_withoutmask_add = self.network_output_g(imageprediction_withoutmask, warped_future_gain,futureframeoffset)  # enhancement to add to predicted frames


                imageprediction_withoutmask_beforeclip = imageprediction_withoutmask + imageprediction_withoutmask_add

                beforeenhancelist.append(imageprediction_withoutmask) #added for error map
                addgainlist.append(imageprediction_withoutmask_add)

                imageprediction_withoutmask_beforeclip = torch.where(imageprediction_withoutmask_beforeclip > 255.0, torch.tensor(255.0).cuda(), imageprediction_withoutmask_beforeclip)
                imageprediction_withoutmask = torch.where(imageprediction_withoutmask_beforeclip < 0.0, torch.tensor(0.0).cuda(), imageprediction_withoutmask_beforeclip)

                predframehistory.append(imageprediction_withoutmask)
                flowhistory.append(flowprediction.clone()) # append predcted flows
                maskhistory.append(futuremask.clone()) # append predict mask


                ## perceptual loss we only compare fused part
                lossname = 'ploss' + str(futureframeoffset)
                target_image_withoutmask = target_image_list[futureframeoffset]

                # fused part perceptual loss, update pixels with future mask
                image_prediction_fused = imageprediction_withoutmask * futuremask + target_image_withoutmask * (1 - futuremask)
                image_prediction_fusedhistory.append(image_prediction_fused)
                target_image = target_image_withoutmask

                imageprediction = image_prediction_fused
                imagepredictionprev = image_prediction_fused.clone()

                if self.lossfunction == 3:  # use l1 loss
                    losses[lossname] = self.L1Loss(imageprediction / self.rgb_max, target_image / self.rgb_max)
             
                b, c, h, w = futuremask.shape #we need mask to be  a lot
                losses["rloss" + str(futureframeoffset)] = 1 - torch.sum(futuremask) / (b*h*w)
                rw = self.costratio
                pw = self.plossratio
                tw = (self.temperalratio ** futureframeoffset)
                bw = self.bceratio

                diff = target_image - imageprediction_withoutmask
                diff = torch.norm(diff / 255.0, p=2, dim=1, keepdim=True)
                gtconfident = torch.where(diff < 0.04, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda()) # 4321
                bceloss = self.bcloss(maskcopy, gtconfident)
                gtconfidentlist.append(gtconfident)

                if futureframeoffset == 0:
                    losses['tot'] = tw * (rw * losses["rloss" + str(futureframeoffset)] + pw * losses["ploss" + str(futureframeoffset)] + bw * bceloss)
                else:
                    losses['tot'] += tw * (rw * losses["rloss" + str(futureframeoffset)] + pw * losses["ploss" + str(futureframeoffset)] + bw * bceloss)




        if self.training:
            return losses, predframehistory, target_image_list
        else:
            ret = {}

            if self.timing:
                torch.cuda.synchronize()
                endtime = time.perf_counter()
                duration = endtime - starttime
                ret["time"] = duration
                print("timed !", duration)

           # ret["losses"] = losses
            ret["image_prediction"] = predframehistory # step is getflow warped image, step is fused , warped image
            ret["flow_prediction"] = flowhistory
            ret["gtimage"] = target_image_list
            ret["inputflows"] = input_flows
            ret["losses"] = losses


            ret["gtconfident"] = gtconfidentlist
            ret["confidentmask"] = maskhistory
            ret["gainlist"] = addgainlist #remove after adding
            ret["beforeenhancelist"] = beforeenhancelist
            if self.phase == "fused":
               ret["fused"] = image_prediction_fusedhistory

            return ret
