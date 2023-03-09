import os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from openslide import OpenSlide
import cv2
import numpy as np
from random import choices
from time import time
import pandas as pd
#%%
ndpisrc = r'\\fatherserverdw\Saurabh\Saurabh\skin\wsi'
dlsrc = r'\\fatherserverdw\kyuex\image\skin_aging\1um\classification_v9_combined'
ecmpatchdst = os.path.join(ndpisrc,'Epi_Patches')
if not os.path.exists(ecmpatchdst):os.mkdir(ecmpatchdst)

#read excel to look up images to process and their patient infos
xlsxpth = r"\\fatherserverdw\Saurabh\Saurabh\skin\wsi\healthy.xlsx"
xlsx = pd.read_excel(xlsxpth,sheet_name='imlist')
xlsx = xlsx[['redcap','Age','Gender']]

dls = [_ for _ in os.listdir(dlsrc) if _.endswith('tif')]
ndpis = [_.replace('tif','ndpi') for _ in dls]

for idx0,(dl,ndpi) in enumerate(zip(dls,ndpis)):
    #load patient info
    redcap = int(os.path.splitext(ndpi)[0])
    pinfo = xlsx[xlsx['redcap']==redcap]
    #load dlmask
    dl = os.path.join (dlsrc,dl)
    dl = Image.open(dl)
    #load wsi
    ndpi = os.path.join(ndpisrc,ndpi)
    wsi = OpenSlide(ndpi)
    #match their sizes
    dlsz = dl.size #dl size
    targetidx = len([_ for _ in wsi.level_dimensions if _[0]>dlsz[0]])-1
    #read wsi at a level closest to dlsize
    start = time()
    wsitarget = wsi.read_region(location=(0,0),level=targetidx,size=wsi.level_dimensions[targetidx]) #read wsi at target level
    end = time()
    print('elapsed time {:.2f}'.format(end-start))
    #resize wsi to match dl size
    wsitarget = np.array(wsitarget)
    wsitarget= cv2.resize(wsitarget, dlsz, interpolation=cv2.INTER_NEAREST)
    wsitarget=wsitarget[:,:,0:3]
    #now we have he and dl with same dimensions
    #we can generate 256x256 patches of ECM based on dlmask
    ECMmask = np.array(dl)
    ECMmask[ECMmask!=2]=0
    #shrink ECMmask to avoid edge
    kernel = np.ones((5, 5), np.uint8)
    ECMmask = cv2.erode(ECMmask, kernel)
    #randomly select k locations to extract ECM patches
    winsz = 32
    #generate 10 for train
    centroids = choices(np.argwhere(ECMmask),k=10)
    images = []
    for centroid in centroids:
        bbox = [centroid[0]-winsz,centroid[0]+winsz,centroid[1]-winsz,centroid[1]+winsz]
        imcrop = wsitarget[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        images.append(imcrop)
    #save patches
    for idx,image in enumerate(images):
        fn = 'train_age{:03d}_patchID{:03d}.png'.format(round(pinfo.Age.iloc[idx]),idx)
        Image.fromarray(image).save(os.path.join(ecmpatchdst,fn))
    #generate 2 for test
    centroids = choices(np.argwhere(ECMmask),k=2)
    images = []
    for centroid in centroids:
        bbox = [centroid[0]-winsz,centroid[0]+winsz,centroid[1]-winsz,centroid[1]+winsz]
        imcrop = wsitarget[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        images.append(imcrop)
    #save patches
    for idx,image in enumerate(images):
        fn = 'test_age{:03d}_patchID{:03d}.png'.format(round(pinfo.Age.iloc[idx]),idx)
        Image.fromarray(image).save(os.path.join(ecmpatchdst,fn))