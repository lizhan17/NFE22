#!/usr/bin/env python
import argparse
import os
import numpy as np
import shutil
import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import pdb
import cv2

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
### masks warning : RuntimeError: Set changed size during iteration #481
# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0
###
import datasets
import models
from utility import tools
from utility.tools import mkdirifnotexist, stddirpath, getmask, getscore
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
#from skimage.measure import compare_psnr, compare_ssim
from sklearn.metrics import precision_score, accuracy_score
from external import flow_vis

import math

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))
###

"""

Fitsum A. Reda, Guilin Liu, Kevin J. Shih, Robert Kirby, Jon Barker, David Tarjan, Andrew Tao, and Bryan Catanzaro. 
"SDC-Net: Video prediction using spatially-displaced convolution.", in ECCV 2018, pp. 718-733. 2018.

"""
parser = argparse.ArgumentParser(description='A PyTorch Implementation of ??')

## blurmask
parser.add_argument('--mywrite', default=1, type=int, help='0 not write image, 1 write image')

parser.add_argument('--memtype', default=1, type=int, help='1 use one 0 use another')

parser.add_argument('--threshold', default=0.4, type=float, help='mask threshold for huber loss')

parser.add_argument('--futurestep', default=3, type=int, help='length of predict step 2=>3')


parser.add_argument('--pr', default=0.3, type=float, help='final image')
parser.add_argument('--br', default=0.3, type=float, help='entroy ')
parser.add_argument('--cr', default=0.3, type=float, help='run time cost ')
parser.add_argument('--tr', default=0.5, type=float, help='temeral ratio default 0.5 ')

parser.add_argument('--timing', default=0, type=int, help='is timing or not default not timing')
parser.add_argument('--phase', default='fused', type=str, help='getflow prediction, getmask for mask, base for baseline')

parser.add_argument('--limgain', default=0, type=int, help='-1 no value, 0 no limitation gain , 1limt gain')

parser.add_argument('--best', default="gpp", type=str, help='gpp is the combined loss score, if psnr we pick best psnr')


parser.add_argument('--mask', default='F', type=str, help='input mask')
parser.add_argument('--inputmasklevel', default=1, type=int, help='0 first level,  1 second level, 2 third level')
parser.add_argument('--lossfunction', default=3, type=int)
parser.add_argument('--model', metavar='MODEL', default='AAA',
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: SDCNet2D)')
parser.add_argument('--save',"--save_root","--s", default='./', type=str, metavar='SAVE_PATH',
                    help='Path of the output folder. (default: current path)')
parser.add_argument('--torch_home', default='./.torch', type=str,
                    metavar='TORCH_HOME',
                    help='Path to store native torch downloads of vgg, etc.')
parser.add_argument('-n', '--name', default='noname', type=str, metavar='RUN_NAME',
                    help='Name of folder for output model')

parser.add_argument('--dataset', default='FrameLoader_ours_cropped_sqtrain', type=str, metavar='TRAINING_DATALOADER_CLASS',
                    help='Specify dataset class for loading (Default: FrameLoader)')

parser.add_argument('--val_dataset', default='FrameLoader_ours_cropped_sqtest', type=str, metavar='Validation_DATALOADER_CLASS',
                    help='Specify dataset class for loading (Default: FrameLoader)')

parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                    help='path to checkpoint (default: none)')

parser.add_argument('--distributed_backend', default='nccl', type=str, metavar='DISTRIBUTED_BACKEND',
                    help='backend used for communication between processes.')

# Resources
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loader workers (default: 4)')
parser.add_argument('-g', '--gpus', type=int, default=1,
                    help='number of GPUs to use')

# Learning rate parameters.
parser.add_argument('--lr', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', default='MultiStepLR', type=str,
                    metavar='LR_Scheduler', help='Scheduler for learning' +
                                                 ' rate (only ExponentialLR, MultiStepLR, PolyLR supported.')
parser.add_argument('--lr_gamma', default=0.1, type=float,
                    help='learning rate will be multipled by this gamma')
parser.add_argument('--lr_step', default=200, type=int,
                    help='stepsize of changing the learning rate')
parser.add_argument('--lr_milestones', type=int, nargs='+',
                    default=[250, 3000], help="Spatial dimension to " +
                                             "crop training samples for training")

# Gradient.
parser.add_argument('--clip_gradients', default=-1.0, type=float,
                    help='If positive, clip the gradients by this value.')

# Optimization hyper-parameters
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='BATCH_SIZE',
                    help='mini-batch per gpu size (default : 4)')
parser.add_argument('--wd', '--weight_decay', default=0.001, type=float, metavar='WEIGHT_DECAY',
                    help='weight_decay (default = 0.001)')
parser.add_argument('--seed', default=1234, type=int, metavar="SEED",
                    help='seed for initializing training. ')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIMIZER',
                    help='Specify optimizer from torch.optim (Default: Adam)')

parser.add_argument('--print_freq', default=100, type=int, metavar="PRINT_FREQ",
                    help='frequency of printing training status (default: 100)')

parser.add_argument('--save_freq', type=int, default=1, metavar="SAVE_FREQ",
                    help='frequency of saving interm1ediate models (default: 20)')

parser.add_argument('--epochs', default=300, type=int, metavar="EPOCHES",
                    help='number of total epochs to run')

# Training sequence, supports a single sequence for now
parser.add_argument('--train_file',  metavar="TRAINING_FILE",
                    help='training file')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 2)')
parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256], metavar="CROP_SIZE",
                    help="Spatial dimension to crop training samples for training (default : [448, 448])")
parser.add_argument('--train_n_batches', default=-1, type=int, metavar="TRAIN_N_BATCHES",
                    help="Limit the number of minibatch iterations per epoch. Used for debugging purposes. \
                    (default : -1")

# FlowNet2 or mixed-precision training experiments

parser.add_argument("--start_index", type=int, default=0, metavar="START_INDEX",
                    help="Index to start loading input data (default : 0)")

# Validation sequence, supports a single sequence for now
parser.add_argument('--val_file', metavar="VALIDATION_FILE",
                    help='validation file (default : None)')
parser.add_argument('--val_batch_size', type=int, default=1,
                    help="Batch size to use for validation.")
parser.add_argument('--val_n_batches', default=-1, type=int,
                    help="Limit the number of minibatch iterations per epoch. Used for debugging purposes.")
parser.add_argument('--video_fps', type=int, default=30,
                    help="Render predicted video with a specified frame rate")
parser.add_argument('--val_freq', default=500000, type=int,
                    help='frequency of running validation')
parser.add_argument('--stride', default=64, type=int,
                    help='The factor for which padded validation image sizes should be evenly divisible. (default: 64)')
parser.add_argument('--initial_eval', action='store_true', default=False,
                    help='Perform initial evaluation before training.')

# Misc: undersample large sequences (--step_size), compute flow after downscale (--flow_scale)
parser.add_argument("--sample_rate", type=int, default=1,
                    help="step size in looping through datasets")
parser.add_argument('--start_epoch', type=int, default=-1,
                    help="Set epoch number during resuming")
parser.add_argument('--skip_aug', action='store_true', help='Skips expensive geometric or photometric augmentations.')

parser.add_argument('--rgb_max', type=float, default=255, help="maximum expected value of rgb colors")

parser.add_argument('--local_rank', default=None, type=int,
                    help='Torch Distributed')

parser.add_argument('--write_images', action='store_true',
                    help='write to folder \'args.save/args.name\' prediction and ground-truth images.')


parser.add_argument('--eval', action='store_true', help='Run model in inference or evaluation mode.')


def parse_and_set_args(block):
    args = parser.parse_args()

    if args.resume != '':
        block.log('setting initial eval to true since checkpoint is provided')
        args.initial_eval = True

    torch.backends.cudnn.benchmark = True
    block.log('Enabling torch.backends.cudnn.benchmark')

    args.rank = int(os.getenv('RANK', 0))
    if args.local_rank:
        args.rank = args.local_rank
    args.world_size = int(os.getenv("WORLD_SIZE", 1))

    args.train_file = stddirpath(args.train_file)
    args.val_file = stddirpath(args.val_file)



    uniquename = ""
    isrerunning = True
    screenlist = ["epochs", "model", "s", "save", "save_root", "train_file", "val_file","resume", "initial_eval"]
    replacedict = {"lossfunction":"L", "sequence_length": "seq", "inputmasklevel": "mlv", "signoidrate": "sgn"}
    if args.name == "noname":  # we ninit a new name
        defaults, input_arguments = {}, {}
        for key in vars(args):
            defaults[key] = parser.get_default(key)
        for argument, value in sorted(vars(args).items()):
            if value != defaults[argument] and argument in vars(parser.parse_args()).keys():
                if argument not in screenlist:
                    if argument in replacedict:
                        argument = replacedict[argument]
                    uniquename = uniquename  + argument + "_"+ str(value) + "-"
                if argument == "model":
                    args.save = os.path.join(args.save, value)
                    os.makedirs(args.save, exist_ok=True)

        args.name = uniquename[:-1]
        isrerunning = False


    args.save_root = os.path.join(args.save, args.name)
    if not isrerunning:
       if os.path.exists(args.save_root):
           print(args.save_root)
           print("loading save root")

    block.log("Creating save directory: {}".format(
        os.path.join(args.save, args.name)))
    os.makedirs(args.torch_home, exist_ok=True)
    os.environ['TORCH_HOME'] = args.torch_home

    defaults, input_arguments = {}, {}
    for key in vars(args):
        defaults[key] = parser.get_default(key)

    savetext = []
    for argument, value in sorted(vars(args).items()):
        if value != defaults[argument] and argument in vars(parser.parse_args()).keys():
            input_arguments['--' + str(argument)] = value
            block.log('{}: {}'.format(argument, value))
            savetext.append('{}: {}'.format(argument, value))


    args.network_class = tools.module_to_dict(models)[args.model]
    args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
    args.dataset_class = tools.module_to_dict(datasets)[args.dataset]
    args.val_dataset_class = tools.module_to_dict(datasets)[args.val_dataset]

    savepath = args.save_root + "log.txt"

    mkdirifnotexist(os.path.dirname(savepath))
    with open(savepath, 'w') as f:
        for item in savetext:
            f.write("%s\n" % (item))
    if args.eval:
        args.train_file = args.train_file

    return args

def initialilze_distributed(args):
    # Manually set the device ids.
    torch.cuda.set_device(args.rank % torch.cuda.device_count())
    # Call the init process
    if args.world_size > 1:
        init_method = 'env://'
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_train_and_valid_data_loaders(block, args):

    # training dataloader
    tkwargs = {'batch_size': args.batch_size,
               'num_workers': args.workers,
               'pin_memory': True, 'drop_last': True}

    train_dataset = args.dataset_class(args,
                                       root=args.train_file, is_training=True)

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        shuffle=(train_sampler is None), **tkwargs)

    block.log('Number of Training Images: {}:{}'.format(
        len(train_loader.dataset), len(train_loader)))

    # validation dataloader
    vkwargs = {'batch_size': args.val_batch_size,
               'num_workers': args.workers,
               'pin_memory': False, 'drop_last': True}

    val_dataset = args.val_dataset_class(args, root=args.val_file)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **vkwargs)

    block.log('Number of Validation Images: {}:{}'.format(
        len(val_loader.dataset), len(val_loader)))

    return train_loader, train_sampler, val_loader


def load_model(model, optimizer, block, args):
    # trained weights

    checkpoint = torch.load(args.resume, map_location='cpu')
    print(checkpoint["state_dict"].keys())

    model_dict = model.state_dict()
    pretrainedict = checkpoint["state_dict"]
    fitletedcheckpoint = {k: v for k, v in pretrainedict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrainedict[k].shape)}

    model.load_state_dict(fitletedcheckpoint, strict=False)


    if 'optimizer' in checkpoint:
        pass
        #optimizer.load_state_dict(checkpoint['optimizer'])  if we freeze some layer it will change for optimizer.

    args.start_epoch = 0 # max(0, checkpoint['epoch'])

def build_and_initialize_model_and_optimizer(block, args):

    model = args.network_class(args)
    block.log('Number of parameters: {}'.format(
        sum([p.data.nelement()
             if p.requires_grad else 0 for p in model.parameters()])))

    block.log('Initializing CUDA')
    assert torch.cuda.is_available(), 'only GPUs support at the moment'
    model.cuda(torch.cuda.current_device())

    optimizer = args.optimizer_class(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    block.log("Attempting to Load checkpoint '{}'".format(args.resume))
    if args.resume and os.path.isfile(args.resume):
        load_model(model, optimizer, block, args)  # here we should not load optilizer
    elif args.resume:
        block.log("No checkpoint found at '{}'".format(args.resume))
        exit(1)
    else:
        block.log("Random initialization, checkpoint not provided.")
        args.start_epoch = 0


    # Run multi-process when it is needed.
    if args.world_size > 1:
        model = DDP(model)

    return model, optimizer

def get_learning_rate_scheduler(optimizer, block, args):
    if args.lr_scheduler == 'MultiStepLR':
        block.log('using multi-step learning rate with {} gamma' +
                  ' and {} milestones.'.format(args.lr_gamma,
                                               args.lr_milestones))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    else:
        raise NameError('unknown {} learning rate scheduler'.format(
            args.lr_scheduler))
    return lr_scheduler

def forward_only(inputs_gpu, model):
    # Forward pass.
    losses, outputs, targets = model(inputs_gpu)
    for k in losses:
        losses[k] = losses[k].mean(dim=0)
    loss = losses['tot']

    return loss, outputs, targets



def evaluate(model, val_loader, block, args, epoch):


    with torch.no_grad():

        loss_values = tools.AverageMeter('loss_values')
        avg_metrics = np.zeros((0, 3), dtype=float)

        # Set the model in evaluation mode.
        model.eval()

        num_batches = len(val_loader) if args.val_n_batches < 0 else args.val_n_batches # hhere batch size is len(validation sets which is 44
        vscorelist1 = []
        vscorelist2 = []
        vscorelist3 = []
        vscorelist4 = []
        totaltimelist = []
        totallosslist = []


        fused_vscorelist1 = []
        fused_vscorelist2 = []
        fused_vscorelist3 = []
        fused_vscorelist4 = []
        fused_psnrlist = []
        rendered_list = []

        seplistpsnr = [[] for i in range(args.futurestep)] # single psnr
        seplistscore1 = [[] for i in range(args.futurestep)] # stat1
        seplistscore2 = [[] for i in range(args.futurestep)] # stat2
        seplistscore3 = [[] for i in range(args.futurestep)] # stat3
        seplistscore4 = [[] for i in range(args.futurestep)] # stat4


        wapseplistpsnr = [[] for i in range(args.futurestep)] # single psnr



        seplist_psnrfused = [[] for i in range(args.futurestep)]  # single psnr
        seplist_rendertime = [[] for i in range(args.futurestep)]  # stat1

        seplist_maskscore = [[] for i in range(args.futurestep)]

    

        for i, batch in enumerate(tqdm(val_loader, total=num_batches)):

            #target_images = batch['image'][-1]
            gt_files_path = batch['input_files'][args.sequence_length:]
            input_files_path = batch['input_files'][:args.sequence_length]
            inputs = {k: [b.cuda() for b in batch[k]] for k in batch if k != 'input_files'}


            ret = model(inputs)

            losses = ret["losses"];
            for k in losses:
                losses[k] = losses[k].mean(dim=0)
            loss = losses['tot']

            output_images = ret["image_prediction"] # enhanced output
            flowpred = ret["flow_prediction"] # list
            target_images = ret['gtimage']


            for s in range(args.futurestep):
                for b in range(args.val_batch_size):
                    pred_image = (output_images[s][b].data.cpu().numpy().transpose(1,2,0) ).astype(np.uint8)
                    gt_image = (target_images[s][b].data.cpu().numpy().transpose(1,2,0) ).astype(np.uint8)

                    gHeight, gWidth, c=  gt_image.shape

                    pred_image = pred_image[:gHeight, :gWidth, :]

                    gt_image = gt_image[:gHeight, :gWidth, :]

                    stats1, stats2, stats3, stats4 = getscore(gt_image, pred_image)
                    psnr = compare_psnr(pred_image, gt_image)

                    if "fused" in ret and args.write_images == True: # we retreive images
                        fusedimage = ret["fused"][s] # final images with masked gt
                        fusedimage = (fusedimage[b].data.cpu().numpy().transpose(1,2,0) ).astype(np.uint8)

                        fusedpsnr = compare_psnr(fusedimage, gt_image)
                        fusedstats1, fusedstats2, fusedstats3, fusedstats4 = getscore(gt_image, fusedimage)

                        if "beforeenhancelist" in ret:
                            wapimagecuda = ret["beforeenhancelist"][s]
                            wapimage = (wapimagecuda[b].data.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)

                            wapimagepsnr = compare_psnr(wapimage, gt_image)
                            wapstats1, wapstats2, wapstats3, wapstats4 = getscore(gt_image, wapimage)
                            wapseplistpsnr.append(wapimagepsnr)

                        fused_psnrlist.append(fusedpsnr)
                        fused_vscorelist1.append(fusedstats1)
                        fused_vscorelist2.append(fusedstats2)
                        fused_vscorelist3.append(fusedstats3)
                        fused_vscorelist4.append(fusedstats4)

                        confidentmask = ret["confidentmask"]
                        confidentmask = (confidentmask[s][b].data.cpu().numpy().transpose(1, 2, 0))


                        rendered_pixelpercentage = 1 - np.sum(confidentmask) / (gHeight * gWidth)
                        rendered_list.append(rendered_pixelpercentage)

                        seplist_rendertime[s].append(rendered_pixelpercentage)
                        seplist_psnrfused[s].append(fusedpsnr)

                    if args.phase == 'fused' and args.write_images == False:

                        stats1 = 1 - loss.data.item() #  we save smallest loss
                        stats2 = 1 - loss.data.item()
                        stats3 = 1 - loss.data.item()
                        stats4 = 1 - loss.data.item()

                    lossvlue = 1 - loss.data.item()
                    totallosslist.append(lossvlue)

                    vscorelist1.append(stats1)
                    vscorelist2.append(stats2)
                    vscorelist3.append(stats3)
                    vscorelist4.append(stats4)

                    seplistpsnr[s].append(psnr)
                    seplistscore1[s].append(stats1)
                    seplistscore2[s].append(stats2)
                    seplistscore3[s].append(stats3)
                    seplistscore4[s].append(stats4)


                    ssim = compare_ssim(pred_image, gt_image, multichannel=True, gaussian_weights=True)
                    err = pred_image.astype(np.float32) - gt_image.astype(np.float32)
                    ie = np.mean(np.sqrt(np.sum(err * err, axis=2)))

                    avg_metrics = np.vstack((avg_metrics, np.array([psnr, ssim, ie])))

                    loss_values.update(loss.data.item(), output_images[s].size(0))

                    if args.rank == 0 and args.write_images:
                        if "time" in ret and s==0:
                            totaltimelist.append(ret["time"])

                        # to do save other images
                        mask1, mask2, mask3, mask4 = getmask(gt_image, pred_image)
                        masklist = [mask1, mask2, mask3, mask4]

                        gtpath = gt_files_path[s][b]
                        gtframename = gt_files_path[-1][b].split("/")[-1].replace(".png","")
                        scenename = gtpath.split("/")[-3]
                        projectname = gtpath.split("/")[-4]
                        savedir = os.path.join(args.save_root, projectname + "_" + scenename +  "_" +  gtframename) #+ "_" + ('%.3f'%psnr) )

                        mkdirifnotexist(savedir)

                        if 'fused' in ret:

                            fusedpath = os.path.join(savedir,  "imagefused_" + str(s) + ".png")
                            if args.mywrite:
                                cv2.imwrite(fusedpath, fusedimage)

                            confidentmask = ret["confidentmask"]
                            maskedpath = os.path.join(savedir, "maskpred_" + str(s) + ".png")
                            maskconfident = (confidentmask[s][b].data.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                            if args.mywrite:
                                cv2.imwrite(maskedpath, 255.0 * maskconfident)


                        predimagepath = os.path.join(savedir,  "imagepred_" + str(s) + ".png") # save pred images
                        if args.mywrite:
                            cv2.imwrite(predimagepath, pred_image)


                        inputimageidex = 0
                        if s == 0:
                            for eachinputpath in input_files_path:
                                src = eachinputpath[0]
                                newpath = os.path.join(savedir, "input_" + str(inputimageidex) + ".png")
                                if args.mywrite:
                                    shutil.copy(src, newpath)
                                inputimageidex += 1

                        ## save gt images
                        newgtpath = os.path.join(savedir, "gtfuture_" + str(s) + ".png")
                        print(gtpath, newgtpath)
                        if args.mywrite:
                            shutil.copy(gtpath, newgtpath)


            if (i + 1) >= num_batches:
                break


        avg_metrics = np.nanmean(avg_metrics, axis=0)
        result2print = 'PSNR: {:.2f}, SSIM: {:.3f}, IE: {:.2f}'.format(
            avg_metrics[0], avg_metrics[1], avg_metrics[2])
        v_psnr, v_ssim, v_ie = avg_metrics[0], avg_metrics[1], avg_metrics[2]
        block.log(result2print)    

    if len(totaltimelist)> 10:
        totaltimelist = totaltimelist[10:] # skip first 10 for warm up
    else:
        totaltimelist = totaltimelist[1:]  # skip first 1 for warm up
    torch.cuda.empty_cache()
    block.log('max memory allocated (GB): {:.3f}: '.format(
        torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    
    return 0

def write_summary(global_index, learning_rate, t_loss, t_loss_avg,
                  v_loss, v_psnr, v_ssim, v_ie, args, epoch, end_of_epoch):
    # Write to tensorboard.
    if args.rank == 0:
        args.logger.add_scalar("train/lr", learning_rate, global_index)
        args.logger.add_scalar("train/trainloss", t_loss, global_index)
        args.logger.add_scalar("train/trainlossavg", t_loss_avg, global_index)
        args.logger.add_scalar("val/valloss", v_loss, global_index)
        args.logger.add_scalar("val/PSNR", v_psnr, global_index)
        args.logger.add_scalar("val/SSIM", v_ssim, global_index)
        args.logger.add_scalar("val/RMS", v_ie, global_index)



def infererence(model, optimizer, lr_scheduler, train_loader,
          train_sampler, val_loader, block, args):

    # Set the model to train mode.
    model.train()

    # Keep track of maximum PSNR.
    max_psnr = -1
    max_score = -1
    # Perform an initial evaluation.
    if args.eval:

        block.log('Running Inference on Model.')
        print("here leng is",len(val_loader))
        evaluate(model, val_loader, block, args, args.start_epoch + 1)

        return 0

    return 0

def main():

    # Initialize torch.distributed.
    with tools.TimerBlock("\nParsing Arguments") as block:
        args = parse_and_set_args(block)

    with tools.TimerBlock("Initializing Distributed"):
        initialilze_distributed(args)

    # Set all random seed for reproducability.
    with tools.TimerBlock("Setting Random Seed"):
        set_random_seed(args.seed)


############################################ flow step #########################################

    with tools.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        train_loader, train_sampler, val_loader = get_train_and_valid_data_loaders(block, args)

    # Build the model and optimizer.
    with tools.TimerBlock("Building {} Model and {} Optimizer".format(
            args.model, args.optimizer_class.__name__)) as block:
        model, optimizer = build_and_initialize_model_and_optimizer(block, args)

    # Learning rate scheduler.
    with tools.TimerBlock("Building {} Learning Rate Scheduler".format(
            args.optimizer)) as block:
        lr_scheduler = get_learning_rate_scheduler(optimizer, block, args)

    # Set the tf writer on rank 0.
    with tools.TimerBlock("Creating Tensorboard Writers"):
        if args.rank == 0 and not args.eval:
            try:
                args.logger = SummaryWriter(logdir=args.save_root)
            except:
                args.logger = SummaryWriter(log_dir=args.save_root)

    # Start training
    with tools.TimerBlock("Inferncing Model") as block:
        infererence(model, optimizer, lr_scheduler, train_loader,
              train_sampler, val_loader, block, args)


    return 0

if __name__ == '__main__':
    torch.cuda.empty_cache() # empty memory
    main()
