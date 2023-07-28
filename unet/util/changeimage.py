from fnmatch import fnmatchcase
from skimage import io
import numpy as np
import glob
import os

def toPng(dir_segmented,dir_GT,dir_png):
  N = 256
  path_seg = sorted(glob.glob(dir_segmented + "/*"))
  path_GT = sorted(glob.glob(dir_GT + "/*"))
  cnt = 0

  for i in range(len(path_seg)):
      img = io.imread(path_seg[i])
      #print(img.shape)
      img_gt = np.zeros((N,N,3),dtype=np.uint8)
      
      for j in range(N):
          for k in range(N):
              img_gt[j,k] = img[j,k+N*2]

      
      for l in range(len(path_GT)):
          img_GT = io.imread(path_GT[l])
          img_GT = np.delete(img_GT,3,axis=2)
          #image = io.imread(path_img[l])
          #print(image)
          #image = np.delete(img_GT,3,axis=2)
          # print(img_img.shape)
          #exit()
          #print(l)
          #print(np.allclose(img_GT, img_gt))
          if np.allclose(img_GT, img_gt):
              # a=path_GT[l]
              # print(a[15:20])
              #exit()
              img_seg = np.zeros((N,N,3),dtype=np.uint8)
              for j in range(N):
                  for k in range(N):
                      img_seg[j,k] = img[j,k+N]
              #print(path_GT[l][-9:])
              #exit()
              io.imsave(dir_png + path_GT[l][-9:],img_seg)
              cnt+=1
              
  #print(cnt)