from PIL import Image
import numpy as np
import glob
import os
from util import image_augmenter as ia
from skimage import io

class Loader(object):
    def __init__(self, dir_original, dir_segmented, init_size=(256, 256), one_hot=True):
        self._data = Loader.import_data(dir_original, dir_segmented, init_size, one_hot)

    def get_all_dataset(self):
        return self._data

    def load_train_test(self, train_rate=0.85, shuffle=True, transpose_by_color=False):
        """
        `Load datasets splited into training set and test set.
         訓練とテストに分けられたデータセットをロードします．
        Args:
            train_rate (float): Training rate.
            shuffle (bool): If true, shuffle dataset.
            transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]
        Returns:
            Training Set (Dataset), Test Set (Dataset)
        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        if transpose_by_color:
            self._data.transpose_by_color()
        if shuffle:
            self._data.shuffle()

        train_size = int(self._data.images_original.shape[0] * train_rate)
        data_size = int(len(self._data.images_original))
        train_set = self._data.perm(0, train_size)
        test_set = self._data.perm(train_size, data_size)
        #print("train_size="+ str(train_size))
        #print("data_size_size="+ str(data_size))

        
        return train_set, test_set

    @staticmethod
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=True):
        # Generate paths of images to load
        # 読み込むファイルのパスリストを作成
        paths_original, paths_segmented = Loader.generate_paths(dir_original, dir_segmented)
        #print("paths_originalはこれです",paths_original)
        #print("paths_segmentedはこれ",paths_segmented)
        # Extract images to ndarray using paths
        # 画像データをndarrayに展開
        images_original, images_segmented = Loader.extract_images(paths_original, paths_segmented, init_size, one_hot)
        #print('data path',dir_original,'   ',dir_segmented)
        print('images_original',images_original.shape)
        print('images_segmented',images_segmented.shape)
        #print(images_segmented[0].shape)
        #io.imsave('イメージセグメントの一枚目.png', images_segmented[0])
        #exit()
        

        # Get a color palette
        # カラーパレットを取得
        image_sample_palette = Image.open(paths_segmented[0])
        print("image_sample_palette.mode",image_sample_palette.mode)
        #image_sample_palette = image_sample_palette.convert("P")
        palette = image_sample_palette.getpalette()
        print("palette",len(palette))
        
        return DataSet(images_original, images_segmented, palette)
        #return DataSet(images_original, images_segmented, palette, augmenter=ia.ImageAugmenter(size=init_size, class_count=len(DataSet.CATEGORY))) # ,augmenter=ia.ImageAugmenter(size=init_size, class_count=len(DataSet.CATEGORY))

    @staticmethod
    def generate_paths(dir_original, dir_segmented):
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        # print(dir_original)
        # print(dir_segmented)
        
        print()
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        #print(filenames)
        # 順番変わる
        # ここでjpgかpngか
        #paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".png", filenames))
        #print(paths_original)
        #print(paths_segmented)
        #exit()
        path = './file.csv'
        with open(path, mode='a') as f:
            for i in range(len(paths_original)):
                f.write(str(i))
                f.write(",")
                f.write(paths_original[i])
                f.write("\n")
            f.write("***********")
            f.write("\n")

        return paths_original, paths_segmented

    @staticmethod
    def extract_images(paths_original, paths_segmented, init_size, one_hot):
        images_original, images_segmented = [], []
        # Load images from directory_path using generator
        print("Loading original images", end="", flush=True)
        for image in Loader.image_generator(paths_original, init_size, antialias=True):
            # image:ndarray
            images_original.append(image)
            if len(images_original) % 100 == 0:
                print("100 original")
                print(".", end="", flush=True)
        print(" Completed", flush=True)
        
        print("Loading segmented images", end="", flush=True)
        for image in Loader.image_generator(paths_segmented, init_size, normalization=False):
            # print(image.dtype) : uint8
            # img = np.copy(image)
            # for i in range(512):
            #     for j in range(512):
            #         if img[i,j] > 1:
            #             img[i,j] = 1
            #images_segmented.append(img)
            images_segmented.append(image)
            if len(images_segmented) % 100 == 0:
                print("100 segmented")
                print(".", end="", flush=True)
        print(" Completed")
        #print(len(images_original),len(images_segmented))
        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        # Change indices which correspond to "void" from 255
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        #print(images_original.shape)
        #print(images_segmented.shape)

        # idee = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
        # print(idee[images_segmented])
        #images_segmented = identity[images_segmented]
        
        # One hot encoding using identity matrix.
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
            print(images_segmented.shape)
            #print(identity)
            images_segmented = identity[images_segmented]
            print("Done")

        else:
            pass
        # image:[data_size][width][height][3]
        # teacher:[data_size][width][height][class]
        return images_original, images_segmented

    @staticmethod
    def cast_to_index(ndarray):
        return np.argmax(ndarray, axis=2)

    @staticmethod
    def cast_to_onehot(ndarray):
        identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
        return identity[ndarray]

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=True, antialias=False,mode=False):
        """
        `A generator which yields images deleted an alpha channel and resized.
         アルファチャネル削除、リサイズ(任意)処理を行った画像を返します
        Args:
            file_paths (list[string]): File paths you want load.
            init_size (tuple(int, int)): If having a value, images are resized by init_size.
            normalization (bool): If true, normalize images.
            antialias (bool): Antialias.
        Yields:
            image (ndarray[width][height][channel]): Processed image
        """
        print(file_paths)
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
                #print(file_path)
                image = Image.open(file_path)
                # print("Image.open(file_path)")
                #print(image.mode)
                # to square
                image = Loader.crop_to_square(image)
                # print("Loader.crop_to_square(image)")
                # print(image)
                
                # resize by init_size
                if init_size is not None and init_size != image.size:
                    if antialias:
                        image = image.resize(init_size, Image.ANTIALIAS)
                    else:
                        image = image.resize(init_size)
                # delete alpha channel
                if mode:
                    image = image.convert("P")
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.asarray(image)
                if normalization:
                    image = image / 255.0
                # １枚ずつ画像を返す
                yield image
                

    @staticmethod
    def crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))


class DataSet(object):

    CATEGORY = (
        "ground",
        "class_0",
        "void"
    )

    """
    CATEGORY = (
        "ground",
        "class_0",
        "class_1",
        "class_2",
        "void"
    )
    """
    '''
    CATEGORY = (
        "ground",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "void"
    )
    '''

    def __init__(self, images_original, images_segmented, image_palette, augmenter=None):
        assert len(images_original) == len(images_segmented), "images and labels must have same length."
        self._images_original = images_original
        self._images_segmented = images_segmented
        self._image_palette = image_palette
        self._augmenter = augmenter
        # print("-------------------------------------------------------------")
        # print(augmenter)
        # print("-------------------------------------------------------------")

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def palette(self):
        return self._image_palette

    @property
    def length(self):
        return len(self._images_original)

    @staticmethod
    def length_category():
        return len(DataSet.CATEGORY)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_original))

    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        return DataSet(images_original, images_segmented, self._image_palette, self._augmenter)

    def shuffle(self):
        idx = np.arange(self._images_original.shape[0])
        np.random.shuffle(idx)
        self._images_original, self._images_segmented = self._images_original[idx], self._images_segmented[idx]

    def transpose_by_color(self):
        self._images_original = self._images_original.transpose(0, 3, 1, 2)
        self._images_segmented = self._images_segmented.transpose(0, 3, 1, 2)


    def perm(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end], self._image_palette,
                       self._augmenter)

    def __call__(self, batch_size=20, shuffle=True, augment=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
         バッチを返すジェネレータです。 デフォルトでバッチはシャッフルされます。
        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size):
            batch = self.perm(start, start+batch_size)
            if augment:
                assert self._augmenter is not None, "you have to set an augmenter."
                yield self._augmenter.augment_dataset(batch, method=[ia.ImageAugmenter.NONE, ia.ImageAugmenter.FLIP])
            else:
                yield batch


if __name__ == "__main__":
    dataset_loader = Loader(dir_original="../data_set/Bone_data_v3/JPEGImages",
                            dir_segmented="../data_set/Bone_data_v3/SegmentationClass")
    train, test = dataset_loader.load_train_test()
    train.print_information()
    test.print_information()
