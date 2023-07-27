import numpy as np

# windowing((array(float)) dcm, (float) wc, (float) ww)
#   dcm   :DICOMデータの画素情報
#          ds = pydicom.dcmread(inputdir)
#          dcm = ds.pixel_array.astype(float)
#   wc    :ウィンドウ中心
#   ww    :ウィンドウ幅
# return (array(float)) image_color
#   image_color :RGBチャネルをもつWindowing後のCT画像

def windowing(dcm, wc, ww):
    Wmin = wc - ww/2
    Wmax = wc + ww/2

    array = dcm
    array = np.where(array < Wmin, Wmin, array)
    array = np.where(array > Wmax, Wmax, array)
    array = array - Wmin
    array_scaled = (array/ww) * 255.0
    array_scaled = np.uint8(array_scaled)

    image = array_scaled.copy()
    image = np.stack([image, image], -1)
    image = np.append(image, image, axis=2)
    image_color = np.split(image, [3], 2)

    return image_color