import os
import subprocess
import time
import pickle
from inspect import isclass
from datetime import timedelta
import numpy as np 
def getscore(gt_image, pred_image, rgb_max=255.0):
        h, w , c = gt_image.shape
        w = float(w)

        stats1 = 1 - (np.sum((np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.04)) / (
                    h * w))
        stats2 = 1 - (np.sum((np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.03)) / (
                    h * w))  # only count pixels
        stats3 = 1 - (np.sum((np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.02)) / (
                    h * w))  # only count pixels
        stats4 = 1 - (np.sum((np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.01)) / (
                    h * w))
        return stats1, stats2, stats3, stats4
        
def getmask(gt_image, pred_image, rgb_max=255.0):
    mask1 = (np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.04)
    mask2 = (np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.03)
    mask3 = (np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.02)
    mask4 = (np.linalg.norm((gt_image / rgb_max - pred_image / rgb_max), axis=2) > 0.01)
    return mask1, mask2, mask3, mask4 

def stddirpath(path):
    if path.endswith("/"):
        path = path[:-1]
    else:
        pass 
    return path 
def mkdirifnotexist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def logger_multithreads(q, prolist, result, freq=5):
    tasknum = len(prolist)
    start = time.time()
    while True:
        if result.ready():
            break
        else:
            size = q.qsize()
            end = time.time()
            passtime = str(timedelta(seconds = end - start))
            totaltime = str(timedelta(seconds = (end - start) / (size * 1.0 / tasknum + 0.00001) ))
            print('%d/%d, %.2f%%, %s, %s'%(size, tasknum, size * 1.0 / tasknum * 100, passtime, totaltime))
            time.sleep(freq)


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.time() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string), flush = True)

def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x))
                 and x not in exclude
                 and getattr(module, x) not in exclude])


# creat_pipe: adapted from https://stackoverflow.com/questions/23709893/popen-write-operation-on-closed-file-images-to-video-using-ffmpeg/23709937#23709937
# start an ffmpeg pipe for creating RGB8 for color images or FFV1 for depth
# NOTE: this is REALLY lossy and not optimal for HDR data. when it comes time to train
# on HDR data, you'll need to figure out the way to save to pix_fmt=rgb48 or something
# similar
def create_pipe(pipe_filename, width, height, frame_rate=60, quite=True):
    # default extension and tonemapper
    pix_fmt = 'rgb24'
    out_fmt = 'yuv420p'
    codec = 'h264'

    command = ['ffmpeg',
               '-threads', '2',  # number of threads to start
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',  # input format
               '-vcodec', 'rawvideo',  # input codec
               '-s', str(width) + 'x' + str(height),  # size of one frame
               '-pix_fmt', pix_fmt,  # input pixel format
               '-r', str(frame_rate),  # frames per second
               '-i', '-',  # The imput comes from a pipe
               '-an',  # Tells FFMPEG not to expect any audio
               '-codec:v', codec,  # output codec
               '-crf', '18',
               # compression quality for h264 (maybe h265 too?) - http://slhck.info/video/2017/02/24/crf-guide.html
               # '-compression_level', '10', # compression level for libjpeg if doing lossy depth
               '-strict', '-2',  # experimental 16 bit support nessesary for gray16le
               '-pix_fmt', out_fmt,  # output pixel format
               '-s', str(width) + 'x' + str(height),  # output size
               pipe_filename]
    cmd = ' '.join(command)
    if not quite:
        print('openning a pip ....\n' + cmd + '\n')

    # open the pipe, and ignore stdout and stderr output
    DEVNULL = open(os.devnull, 'wb')
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)

# AverageMeter: code from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


