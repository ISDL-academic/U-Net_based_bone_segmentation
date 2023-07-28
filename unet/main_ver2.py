#https://tensorflow.classcat.com/category/semantic-segmentation/
import argparse
import random
#import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import glob
import numpy as np
#import pydicom
from util import loader as ld
from util import model_ver2 as model
from util import repoter as rp
from util import changeimage as ci
import os
from skimage import io
from util import iou as iu
import csv
from statistics import stdev, variance

# Dataset path
spine = "Spine/256"
femur = "Femur/256"
bone = spine
print("data:",bone)

def load_dataset(train_rate):
    # original  image
    # segmented GT
    dir_original="data_set/" + bone + "/train_images"
    dir_segmented="data_set/" + bone + "/train_GT_images"
    loader = ld.Loader(dir_original, dir_segmented)
    return loader.load_train_test(train_rate=train_rate, shuffle=False)

def load_dataset_eval(train_rate):
    # original  image
    # segmented GT
    dir_original="data_set/" + bone + "/removed_even"
    dir_segmented="data_set/" + bone + "/removed_GT_even"
    loader = ld.Loader(dir_original, dir_segmented)
    return loader.load_train_test(train_rate=train_rate, shuffle=False)

def set_seed(seed=0):
    tf.set_random_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def train(parser):
    # Load train and test datas
    train, valid = load_dataset(train_rate=parser.trainrate)
    
    print("train:",train.images_original.shape)
    print("valid:",valid.images_original.shape)
    print(parser)
    trainrate = 0.001
    print("trainrateは",trainrate,"です")
    print("train:",train.images_original.shape)
    print("valid:",valid.images_original.shape)
    print(train(batch_size=parser.batchsize, augment=parser.augmentation))
    
    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "valid"]) # create gragh 
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "valid"])
    iou_fig = reporter.create_figure("IoU", ("epoch", "IoU"), ["train", "valid"])

    # Whether or not using a GPU
    gpu = parser.gpu

    # Create a model
    # l2reg', type=float, default=0.0001
    set_seed(0)
    model_unet = model.UNet(l2_reg=parser.l2reg).model
    print(parser.l2reg)

    print("model_unet.inputs IS ",model_unet.inputs)
    print("model_unet.outputs IS ",model_unet.outputs)
    print("model_unet.teacher IS ",model_unet.teacher)
    print("model_unet.is_training IS",model_unet.is_training)
    # Set a loss function and an optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(trainrate).minimize(cross_entropy)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    #accuracy,update_op=tf.metrics.accuracy(model_unet.inputs,model_unet.teacher)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy IS",accuracy)

    
    #Calculate IOU
    #https://ai-pool.com/d/iou_by_tensorflow
    #https://www.sigfoss.com/developer_blog/detail?actual_object_id=147
    I = tf.reduce_sum(tf.argmax(model_unet.outputs, 3) * tf.argmax(model_unet.teacher, 3))
    U = tf.reduce_sum(tf.argmax(model_unet.outputs, 3) + tf.argmax(model_unet.teacher, 3)) - I
    iou = tf.reduce_mean(I / U)
    
    # Initialize session
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},log_device_placement=False, allow_soft_placement=True)
    server=tf.train.Saver()
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train the model
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation
    train_dict = {model_unet.inputs: train.images_original, model_unet.teacher: train.images_segmented,
                  model_unet.is_training: False}
                  
    valid_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                 model_unet.is_training: False}
                 
                 

    
    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):

            inputs = batch.images_original
            #print("inputs IS ",inputs.shape)
            teacher = batch.images_segmented
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})
                                            

        # Evaluation
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_valid = sess.run(cross_entropy, feed_dict=valid_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_valid = sess.run(accuracy, feed_dict=valid_dict)
            
            iou_train = sess.run(iou, feed_dict=train_dict)
            iou_valid = sess.run(iou, feed_dict=valid_dict)
            
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train , "IoU:", iou_train)
            
            # out = sess.run(model_unet.outputs,feed_dict=train_dict)
            # print("model_unet.outputs ARE ",out.shape)
            # exit()

            accuracy_fig.add([accuracy_train, accuracy_valid], is_update=True)
            loss_fig.add([loss_train, loss_valid], is_update=True)
            iou_fig.add([iou_train, iou_valid], is_update=True)
            if epoch % 1 == 0:
                #idx_train = random.randrange(10)
                #idx_test = random.randrange(100)
                
                idx_train = random.randrange(len(train.images_original))
                idx_valid = random.randrange(len(valid.images_original))
                
                outputs_train = sess.run(model_unet.outputs,feed_dict={model_unet.inputs: [train.images_original[idx_train]],model_unet.is_training: False})
                outputs_test  = sess.run(model_unet.outputs,feed_dict={model_unet.inputs: [valid.images_original[idx_valid]],model_unet.is_training: False})
                #print("outputs_train IS",outputs_train.shape)
                
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                valid_set = [valid.images_original[idx_valid], outputs_test[0], valid.images_segmented[idx_valid]]
                #print("train_set IS",np.array(train_set).shape)
                #exit()
                no_valid = len(valid.images_original)+idx_valid+1
                reporter.save_image_from_ndarray(train_set, valid_set, train.palette, epoch, idx_train, no_valid, index_void=len(ld.DataSet.CATEGORY)-1)
    
    reporter.save_model(server,sess)
    
    # Test the trained model
    eval_train, eval_test = load_dataset_eval(1.0)
    for i in range(len(eval_train.images_original)):
        outputs_train = sess.run(model_unet.outputs,
                            feed_dict={model_unet.inputs: [eval_train.images_original[i]],
                                        model_unet.is_training: False})
        train_set = [eval_train.images_original[i], outputs_train[0], eval_train.images_segmented[i]]
        reporter.save_image_from_ndarray(train_set, train_set, train.palette, 10000 , i, i, index_void=len(ld.DataSet.CATEGORY)-1)

                                    
    I = tf.reduce_sum(tf.argmax(model_unet.outputs, 3) * tf.argmax(model_unet.teacher, 3))
    U = tf.reduce_sum(tf.argmax(model_unet.outputs, 3) + tf.argmax(model_unet.teacher, 3)) - I
    iou = tf.reduce_mean(I / U)
    test_dict = {model_unet.inputs: eval_train.images_original, model_unet.teacher: eval_train.images_segmented,model_unet.is_training: False}
                 
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
   
    # Convert to a single image　and calculate IoU per image
    if bone == femur:
      os.makedirs(reporter._result_dir + "/image/train_output",exist_ok=True)
      os.makedirs(reporter._result_dir + "/image/test_output",exist_ok=True)
      ci.toPng(dir_segmented=reporter._image_train_dir ,dir_GT="data_set/" + bone + "/train_GT_images",dir_png = reporter._result_dir + "/image/train_output/")
      ci.toPng(dir_segmented=reporter._image_test_dir ,dir_GT="data_set/" + bone + "/removed_GT_even",dir_png = reporter._result_dir + "/image/test_output/")
      path_GT = sorted(glob.glob("data_set/" + bone + "/removed_GT_even/*"),key=iu.natural_keys)

    elif bone == spine:
      os.makedirs(reporter._result_dir + "/image/train_output",exist_ok=True)
      os.makedirs(reporter._result_dir + "/image/test_output",exist_ok=True)
      ci.toPng(dir_segmented=reporter._image_train_dir ,dir_GT="data_set/" + bone + "/train_GT_images",dir_png = reporter._result_dir + "/image/train_output/")
      ci.toPng(dir_segmented=reporter._image_test_dir ,dir_GT="data_set/" + bone + "/removed_GT_even",dir_png = reporter._result_dir + "/image/test_output/")
      path_GT = sorted(glob.glob("data_set/" + bone + "/removed_GT_even/*"),key=iu.natural_keys)

    path_train = sorted(glob.glob(reporter._result_dir + "/image/train_output/*"),key=iu.natural_keys)
    path_test = sorted(glob.glob(reporter._result_dir + "/image/test_output/*"),key=iu.natural_keys)
    
    #print(path_seg)
    #print(path_GT)
    iou_all = []
    with open(reporter._result_dir + "/IoU.csv", "w", newline="") as f:
      for i in range(len(path_test)):
        img = io.imread(path_test[i])
        img_GT = io.imread(path_GT[i])
        iou_all.append(iu.calcIoU(img,img_GT))
        header_row = [path_test[i][-9:],iu.calcIoU(img,img_GT)]
        writer = csv.writer(f)
        writer.writerow(header_row)
      print("IoU:",sum(iou_all)/len(iou_all))
      print("IoU var:",variance(iou_all))
    #reporter.save_model(server,sess)
    sess.close()
    
    

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )


    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.8, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('-m', '--model', type=str, default="aspp", help='aspp or u-net')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    # 乱数シードを固定する
    set_seed(0)
    train(parser)
