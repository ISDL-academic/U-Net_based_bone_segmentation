# -*- coding: utf-8 -*-
#部位ごとにiouをとってみようs
#from msilib.sequence import tables
import sys
import numpy as np
from PIL import Image
from skimage import io
import csv
import glob, re
from statistics import stdev, variance

#color_base = [[0,0,0],[128,0,0],[0,128,0]]
c = 255
color_io = [[0,0,0],[c,c,0]]
N_clusters = len(color_io)

def setLabel(estimated_img,true_img,num):
  # Set pred data-----
  # Set Ground-truth data-----
  height, width = estimated_img.shape[:2]
  no = np.zeros([height*width, ])
  P = np.zeros([height*width, 4])
  cnt = 0
  for i in range(height):
    for j in range(width):
      tmp_no_label = estimated_img[i,j]
      tmp_gt_label = true_img[i,j][:3]

      tmp_list_no_label =  tmp_no_label.tolist()
      tmp_list_gt_label =  tmp_gt_label.tolist()

      flag = 0
    
      for k in range(N_clusters): #2
        if (color_io[k]==tmp_list_no_label):#black 0 yellow 1 
          no_label = k
          flag = 1
        if flag == 0 :
          no_label=0

        if (color_io[k]==tmp_list_gt_label):
          gt_label = k
          flag = 1
        if flag == 0 :
          gt_label=0

      no[cnt] = no_label
      P[cnt,0]=i
      P[cnt,1]=j
      P[cnt,2]=num
      P[cnt,3]=gt_label
      cnt+=1

  return no, P

def score(estmated_label, train_P, N_clusters):
    list_estimated = np.zeros([len(estmated_label),4])
    list_gt = np.zeros([len(estmated_label),4])

    for i in range(len(estmated_label)):
        list_estimated[i,0] = estmated_label[i]
        list_estimated[i,1] = train_P[i,0]
        list_estimated[i,2] = train_P[i,1]
        list_estimated[i,3] = train_P[i,2]

        list_gt[i,0] = train_P[i,3]
        list_gt[i,1] = train_P[i,0]
        list_gt[i,2] = train_P[i,1]
        list_gt[i,3] = train_P[i,2]

    list =  [i for i in range(N_clusters)]

    tuple_list =  [0 for i2 in range(N_clusters)]
    tuple_list_gt =  [0 for i2 in range(N_clusters)]
    index =  [0 for i2 in range(N_clusters)]
    index_gt =  [0 for i2 in range(N_clusters)]
    after_list = []
    after_list_gt = []

    for i in range(int(N_clusters)):
        index[i] = np.where(list_estimated[:, 0] == i)
        index_gt[i] = np.where(list_gt[:, 0] == i)

        after_list.append(list_estimated[index[i]])
        after_list_gt.append(list_estimated[index_gt[i]])

    for i in range(int(N_clusters)):
        tuple_list[i] =  map(tuple, after_list[i][:,1:4])
        tuple_list_gt[i] =  map(tuple, after_list_gt[i][:,1:4])
        tuple_list[i] =set(tuple_list[i])
        tuple_list_gt[i] =set(tuple_list_gt[i])

   
    tp = len(tuple_list[1].intersection(tuple_list_gt[1]))
    fp = len(tuple_list[1].difference(tuple_list_gt[1]))
    fn = len(tuple_list_gt[1].difference(tuple_list[1]))
    tp_fp_fn = len(tuple_list[1].union(tuple_list_gt[1]))
    # print(fp+tp+fn)
    # print(tp_fp_fn)
    # print(tp)
    # print(fn)
    return tp,fp,fn

# estimated_img:セグメンテーション結果
# true_img:GT画像
#   return  tp/(tp+fp+fn)
#  tp/(tp+fp+fn):tp,fp,fnからIoUを計算
def calcIoU(estimated_img,true_img):
  height, width = estimated_img.shape[:2]
  est = np.zeros([height*width, ])
  P = np.zeros([height*width, 4])
  N_clusters = len(color_io)
  num=0
  no,P = setLabel(estimated_img,true_img,num)
  tp,fp,fn = score(no, P, N_clusters)
  return tp/(tp+fp+fn)
