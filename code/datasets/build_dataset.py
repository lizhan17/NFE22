import os
import glob
import tqdm
import numpy as np
import pickle
import multiprocessing
from datasets.dataset_utils import StaticRandomCrop
from datasets.dataset_utils import getflowfromimagepath
import cv2


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def movetarfiletodir():
    tarfilelist = glob.glob("/mnt/2t/docker_containers/tmp/debug/*/*/*.tar.gz")
    for eachtarfile in tarfilelist:
        dir = os.path.basename(eachtarfile).replace(".tar.gz","")
        dstdir = os.path.join( "/mnt/2t/docker_containers/tmp/debug/sample/", dir)
        print(dstdir)
        if not os.path.exists(dstdir):
            os.mkdir(dstdir)
        dstpath = os.path.join(dstdir, os.path.basename(eachtarfile))
        print(dstpath)
        #os.rename(eachtarfile, dstpath)

def untareachfile():
    tarfilelist = glob.glob("/mnt/2t/docker_containers/tmp/debug/*/*.tar.gz")
    for eachtar in tarfilelist:
        print(eachtar)
        cmd = "tar -xzf " + eachtar + " -C " + os.path.dirname(eachtar)
        os.system(cmd)

def unzipfile():
    zipfilelist = glob.glob("/mnt/8t/docker_containers/tartan/tartanair_tools/downloaded/*/Easy/flow_mask.zip")
    #zipfilelist = glob.glob("/mnt/8t/docker_containers/tartan/tartanair_tools/downloaded/westerndesert/Easy/*.zip")
    for eachzip in zipfilelist:

        cmd = "unzip -o " + eachzip + " -d " + "/mnt/8t/docker_containers/tartan/train/" #"#os.path.dirname(eachzip)
        print(cmd)
        os.system(cmd)

# def checktantar():
#     root = "/mnt/2t/docker_containers/tmp/debug/sample/*/*/image_left/*_left.png"
#     inputimagepaths = glob.glob(root)
#     print(len(inputimagepaths))
#
#     for eachpngfile in inputimagepaths:
#         imagedir = os.path.dirname(eachpngfile)
#         basename = os.path.basename(eachpngfile)
#
#         assert basename.endswith("_left.png")
#         newbasename = basename.replace("_left.png", "")
#         baseid = int(newbasename)
#         nextbasename = str(baseid + 1).zfill(len(newbasename))
#         flowname = newbasename + "_" + nextbasename + "_flow.npy"
#         flowdir = imagedir.replace("image_left", "flow")
#         flowpath = os.path.join(flowdir, flowname)
#         try:
#             assert os.path.exists(flowpath)
#         except:
#             print(flowpath)

def rmfile():
    tarfilelist = glob.glob("/mnt/2t/docker_containers/tmp/debug/*/*/*.tar.gz")
    print(len(tarfilelist))
    for file in tarfilelist:
        cmd = "rm " + file
        #os.system(cmd)

def collect_filelist(self, root):
    include_ext = [".png", ".jpg", "jpeg", ".bmp"]
    # collect subfolders, excluding hidden files, but following symlinks
    #
    print(root)

    dirs = glob.glob(root + "/*/*/image_left/")
    assert (len(dirs) > 1)
    dirs = natsort.natsorted(dirs)

    # naturally sort, both dirs and individual images, while skipping hidden files
    datasets = []
    for eachdir in dirs:
        pngs = os.listdir(eachdir)
        pngs = natsort.natsorted(pngs)
        pngs = pngs[:-1]
        datasets.append(pngs)

    # datasets = [
    #     [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
    #      if os.path.isfile(os.path.join(fdir, el))
    #      and not el.startswith('.')
    #      and any([el.endswith(ext) for ext in include_ext])]
    #     for fdir in dirs
    # ]

    return [el for el in datasets if el]


def getflowfromimagepath(inputimagepaths, flowfiletype):
    inputflowpaths = []
    if "tantar" in flowfiletype:
        for eachpngfile in inputimagepaths:
            imagedir = os.path.dirname(eachpngfile)
            basename = os.path.basename(eachpngfile)

            assert basename.endswith("_left.png")
            newbasename = basename.replace("_left.png","")
            baseid = int(newbasename)
            nextbasename = str(baseid + 1).zfill(len(newbasename))
            flowname = newbasename + "_" + nextbasename + "_flow.npy"
            flowdir = imagedir.replace("image_left","flow")
            flowpath = os.path.join(flowdir, flowname)
            try:
                assert os.path.exists(flowpath)
            except:
                print(flowpath)

        pass ##todo
    return inputflowpaths

def checkimages():
    root = "/mnt/8t/docker_containers/tartan/train/*/*/Easy/*/image_left/*_left.png"
    pnglist = glob.glob(root)

    getflowfromimagepath(pnglist, "tantar")


import numpy as np
import os

TAG_FLOAT = 202021.25

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

def stats(para):
    flowfile, flowtype, q = para
    if flowtype == "tartan":
        savepath = flowfile.replace(".npy",".stat")
        if os.path.exists(savepath):
            pass
        else:
            flowdata = np.load(flowfile)
            flowdata = np.abs(flowdata)
            h,w,c = flowdata.shape
            flowdata[:, :, 0] = flowdata[:,:,0] / w
            flowdata[:, :, 1] = flowdata[:, :, 1] / h
            maxvalue = np.amax(flowdata)
            minvalue = np.amin(flowdata)
            hist, bin_edges = np.histogram(flowdata, bins=[0,0.01,0.02, 0.03,0.04,0.05,0.1,99999])
            hist = hist / w / h / 2
            dict = {"max": maxvalue, "min": minvalue, "hist": hist}
            save_pkl(savepath , dict)
            dict.clear()
    if flowtype == "sintel":
        try:
            savepath = flowfile.replace(".flo", ".stat")
            # if os.path.exists(savepath):
            #     pass
            # else:
            flowdata = readFlowFile(flowfile)
            flowdata = np.abs(flowdata)
            h, w, c = flowdata.shape
            flowdata[:, :, 0] = flowdata[:, :, 0] / w
            flowdata[:, :, 1] = flowdata[:, :, 1] / h
            maxvalue = np.amax(flowdata)
            minvalue = np.amin(flowdata)
            hist, bin_edges = np.histogram(flowdata, bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 99999])
            hist = hist / w / h / 2
            dict = {"max": maxvalue, "min": minvalue, "hist": hist}
            save_pkl(savepath, dict)
            dict.clear()
        except:
            print("errr")

    if q is not None:
        q.put(0)


def getstats():
    # get stats for


    dataset_root = "/mnt/8t/opticalflowdata/FlyingChairs_release/data/*.flo" #'/mnt/2t/data/sintel_mpi/*/flow/*/*.flo'
    p = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    q = m.Queue()

    imlist = glob.glob(dataset_root)
    print('got file list length :', len(imlist))

    prolist = []
    for impath in tqdm.tqdm(imlist):
        prolist.append([impath, "sintel", q])
    result = p.map_async(stats, prolist)
    logger_multithreads(q, prolist, result)

def parse_stat():
    import matplotlib.pyplot as plt
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3,sharey=True)

    # fig, ax1 = plt.subplots()

    root = '/mnt/2t/data/sintel_mpi/*/flow/*/*.stat'#'/mnt/8t/docker_containers/tartan/train/*/*/Easy/*/flow/*.stat'
    biglist = [0,0,0,0,0,0,0]

    import numpy as np
    cnt = 0
    biglist = np.asarray(biglist).astype("float64")
    for eachfile in tqdm.tqdm(glob.glob(root)):

        loaddata = load_pkl(eachfile)
        biglist += np.asarray(loaddata["hist"])
        # if projectname not in bigditct:
        #     bigditct[projectname] = [biggerthan2]
        # else:
        #     bigditct[projectname].append(biggerthan2)
        cnt += 1
    # hist, bin_edges = np.histogram(biglist, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1])
    # print(hist)
    # print(sum(hist))
    biglist = biglist /cnt


    plt.ylabel('pixel percentage')
    name_list = ["Flow0-1%","Flow 1-2%", "Flow 2-3%","Flow 3-4%","Flow 4-5%","Flow 5-10%","Flow > 10%"]
    xval = [1,2,3,4,5,6,7]
    cnt= 0
    for j in range(len(biglist)):
        cnt +=1
        ax2.bar(xval[j], biglist[j], width=0.8, bottom=0.0, align='center', color="b", alpha=0.13*cnt,
                label=name_list[j])


###
    root = '/mnt/8t/docker_containers/tartan/train/*/*/Easy/*/flow/*.stat'  # '/mnt/8t/docker_containers/tartan/train/*/*/Easy/*/flow/*.stat'
    biglist = [0, 0, 0, 0, 0, 0, 0]

    import numpy as np
    cnt = 0
    biglist = np.asarray(biglist).astype("float64")
    for eachfile in tqdm.tqdm(glob.glob(root)):
        projectname = eachfile.split("/")[-5]
        subproject = eachfile.split("/")[-3]
        filename = eachfile.split("/")[-1][0:6]
        loaddata = load_pkl(eachfile)
        biglist += np.asarray(loaddata["hist"])

        cnt += 1

    biglist = biglist / cnt

    xval = [7 + e for e in [1, 2, 3, 4, 5, 6, 7]]
    cnt = 0
    for j in range(7):
        cnt += 1
        ax1.bar(xval[j], biglist[j], width=0.8, bottom=0.0, align='center', color="b", alpha=0.13 * cnt,
                label=name_list[j])

####

    root =  "/mnt/8t/opticalflowdata/FlyingChairs_release/data/*.stat"   # '/mnt/8t/docker_containers/tartan/train/*/*/Easy/*/flow/*.stat'
    biglist = [0, 0, 0, 0, 0, 0, 0]

    import numpy as np
    cnt = 0
    biglist = np.asarray(biglist).astype("float64")
    for eachfile in tqdm.tqdm(glob.glob(root)):
        loaddata = load_pkl(eachfile)
        biglist += np.asarray(loaddata["hist"])

        cnt += 1

    biglist = biglist / cnt

    xval = [14 + e for e in [1, 2, 3, 4, 5, 6, 7]]
    cnt = 0
    for j in range(7):
        cnt += 1
        ax3.bar(xval[j], biglist[j], width=0.8, bottom=0.0, align='center', color="b", alpha=0.13 * cnt,
                label=name_list[j])

####
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    plt.xlabel('flow distribution')
    ax1.title.set_text('Tartan')
    ax2.title.set_text('Sintel')
    ax3.title.set_text('FlyingChair')



####
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels,loc=(0.88,0.62))
    # frame1 = plt.gca()
    #
    ax1.get_xaxis().set_visible(False)
    #
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)

    fig.text(0.54, 0.01, 'Flow Distribution', ha='center',fontsize=15)
    fig.text(0.02, 0.5, 'Pixel Percentage', va='center', rotation='vertical',fontsize=15)

    fig.tight_layout()
    fig.set_size_inches(15, 6)

    plt.show()

def flowwithmask():
    root = "/mnt/8t/docker_containers/tartan/train/*/*/Easy/*/flow/*_flow.npy"
    masklist= glob.glob(root)
    for eachmask in tqdm.tqdm(masklist):
        flow = eachmask.replace("_flow.npy","_mask.npy")
        if not os.path.exists(flow):
            print(flow)

def sepframe():
    pass
def removesintel():
    root = "/mnt/8t/docker_containers/sintelvalid/final/*/*.png"
    pnglist = glob.glob(root)
    print(len(pnglist))
    # for png in pnglist:
    #     basename = os.path.basename(png).replace(".png","")
    #     idname = basename[-4:]
    #     if int(idname) > 5:
    #         print(png)
    #         os.remove(png)


# def buildataset():
#     root =

import random
def movesdd():
    root = "/mnt/8t/docker_containers/tartan/train/*/*/Easy/*"
    listproject = os.listdir( "/mnt/8t/docker_containers/tartan/train/")
    print(listproject)
    import shutil
    for eachp in tqdm.tqdm(listproject):
        destp = os.path.join("/mnt/8t/docker_containers/tartan/train/", eachp)
        folders = glob.glob(destp + "/*/Easy/P*")
        number = int(len(folders) /2)
        for eachfolder in folders[:number]:
            src = eachfolder
            dest = eachfolder.replace("/mnt/8t","/mnt/2t")
            destdir = os.path.dirname(dest)
            mkdirifnotexist(destdir)
            cmd = "cp -r " + src + " " + dest
            print(cmd)
            os.system(cmd)
def testslow():
    import natsort

    def getflowfromimagepath(inputimagepaths, flowfiletype):
        inputflowpaths = []
        if "tantar" in flowfiletype:
            for eachpngfile in inputimagepaths:
                imagedir = os.path.dirname(eachpngfile)
                basename = os.path.basename(eachpngfile)

                assert basename.endswith("_left.png")
                newbasename = basename.replace("_left.png", "")
                baseid = int(newbasename)
                nextbasename = str(baseid + 1).zfill(len(newbasename))
                flowname = newbasename + "_" + nextbasename + "_flow.npy"
                flowdir = imagedir.replace("image_left", "flow")
                flowpath = os.path.join(flowdir, flowname)

                assert os.path.exists(flowpath)

                flowdata = np.load(flowpath, allow_pickle=False)
                print(flowdata.shape)
                inputflowpaths.append(flowdata)
        if "sintel" in flowfiletype:
            for eachpng in inputimagepaths:
                basename = os.path.basename(eachpng)
                assert basename.startswith(("frame_"))
                flowpath = eachpng.replace(".png", ".flo").replace("/final", "/flow")
                assert os.path.exists(flowpath)
                flowdata = readFlowFile(flowpath)
                inputflowpaths.append(flowdata)

        return inputflowpaths

    sequence_length = 2
    root = "/drepo/tartan/train"#"/drepo/tartan/train"
    if "debug" not in root:
        dirs = glob.glob(root + "/*/*/Easy/P*/image_left/")  #
    else:
        dirs = glob.glob(root + "/*/*/image_left/")  #

    assert (len(dirs) > 1)
    dirs = natsort.natsorted(dirs)

    # natura
    # lly sort, both dirs and individual images, while skipping hidden files


    import time
    import random
    #random.shuffle(datasets)
    start = time.time()
    for i in range(10):
        idx= random.randint(0,len(datasets))
        f = datasets[idx]
        getflowfromimagepath(f,"tantar")
    print("cost", time.time() -start)

    #datasets = datasets[:2]# only 10 datasets
    # datasets1 = [['/drepo/tartan/train/office2/office2/Easy/P004/image_left/000292_left.png', '/drepo/tartan/train/office2/office2/Easy/P004/image_left/000293_left.png', '/drepo/tartan/train/office2/office2/Easy/P004/image_left/000294_left.png']]
    # datasets2 = [['/drepo/tmp/debug/sample/amusement_sample_P008/P008/image_left/000415_left.png', '/drepo/tmp/debug/sample/amusement_sample_P008/P008/image_left/000416_left.png', '/drepo/tmp/debug/sample/amusement_sample_P008/P008/image_left/000417_left.png']]
    # start = time.time()
    # for eachseq in datasets1:
    #     f = getflowfromimagepath(eachseq,"tantar")



def single_build(arg):
    input_files, crop_size, sequenlenth, flowtype, q = arg
    images = [cv2.imread(imfile) for imfile in input_files]
    input_shape = images[0].shape[:2]

    flows = getflowfromimagepath(input_files[:-2], flowtype)
    print("d")

    cropper = StaticRandomCrop(crop_size, input_shape)
    print("b")
    images = map(cropper, images)
    input_shape = crop_size  # zhan added
    flows = map(cropper, flows)


    input_images = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in images]
    gtflows = [torch.from_numpy(flow.transpose(2, 0, 1)).float() for flow in flows]

    output_dict = {
        'image': input_images, 'ishape': input_shape, 'input_files': input_files, "gt_flow": gtflows
    }
    del images
    del flows
    savepath = input_files[0].replace(".png", "_x"+str(sequenlenth)+"_"+str(crop_size) + ".pkl")
    save_pkl(savepath, output_dict)
    print(savepath)
    if q is not None:
        q.put(0)

def builddataset():
    import natsort
    import torch
    import cv2

    def collect(root, sequence_length):
        if "debug" not in root:
            dirs = glob.glob(root + "/*/*/Easy/P*/image_left/")  #
        else:
            dirs = glob.glob(root + "/*/*/image_left/")  #

        assert (len(dirs) > 1)
        dirs = natsort.natsorted(dirs)
        datasets = []
        # create sequences
        for eachdir in dirs:
            pngs = os.listdir(eachdir)
            pngs = natsort.natsorted(pngs)
            pngs = pngs[:-sequence_length]  # ignore last frame
            for basename in pngs:
                assert basename.endswith("_left.png")
                newbasename = basename.replace("_left.png", "")
                baseid = int(newbasename)
                invalid = False
                sequence = []
                for j in range(1, sequence_length + 2):
                    currentname = str(baseid + j - 1).zfill(len(newbasename))
                    nextbasename = str(baseid + j).zfill(len(newbasename))
                    flowname = currentname + "_" + nextbasename + "_flow.npy"
                    flowdir = eachdir.replace("image_left", "flow")
                    flowpath = os.path.join(flowdir, flowname)
                    if not os.path.exists(flowpath):
                        invalid = True
                    currentimagepath = os.path.join(eachdir, currentname + "_left.png")
                    nextimagepath = os.path.join(eachdir, nextbasename + "_left.png")
                    if not os.path.exists(nextimagepath):
                        invalid = True
                    sequence.append(currentimagepath)
                if invalid == True:
                    pass
                else:
                    datasets.append(sequence)
        return  datasets
    targetroot = "/drepo/tmp/debug/sample"
    flowtype = "tantar"
    sequence_length = 2
    crop_size = 256

    p = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    q = m.Queue()

    prolist = []


    seqlist = collect(targetroot, sequence_length)
    print("build from ",len(seqlist))
    for seq in tqdm.tqdm(seqlist):
        prolist.append([seq,  crop_size, sequence_length,flowtype,q])
    result = p.map_async(single_build, prolist)
    logger_multithreads(q, prolist, result)




if __name__ == '__main__':
    #rmfile()
    #movetarfiletodir()
    #checkimages()
    #unzipfile()
    #getstats()
    #parse_stat()
    #flow()
    #flowwithmask()
    #removesintel()
    #movesdd()
    builddataset()

