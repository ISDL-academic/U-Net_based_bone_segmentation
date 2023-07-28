# myResearch:U-Net based bone segmentation with ASPP and CRF 

## U-Net
You can excute U-Net with unet/Unet.ipynb


## Data pre-processing
### Optimizs Windowing Parameter with jDE
You can optimize windwowing parameter,which is the part of dicom parameter,using windowPeocessing/window_jDE/main.py.
To handle dicom data, you need a libraty called pydicom, so please install that.

The fitness function of jDE is IoU between optimized images and Ground Truth images.

<!--
windowPeocessing/window_jDE/main.pyでwindow parameterをjDEで最適化できます  
pydicomが必要なのでインストールしてください

iou.pyのcalcIoU関数でIoUを計算、windowing.pyのwindowing関数でwindow処理をしています
-->
