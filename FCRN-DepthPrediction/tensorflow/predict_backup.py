# @Author: Hao G <hao>
# @Date:   2018-01-12T12:28:46+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-12T17:34:09+00:00



import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import ipdb
import handy_function as hf
import cv2

import models

def predict(model_data_path, image_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img_array = Image.open(image_path)
    img_array = img_array.rotate(-90, expand=True)
    [height_old, width_old, _] = np.asarray(img_array).shape
    img = img_array.resize((width, height), Image.BICUBIC)
    img = np.asarray(img)

    # # img_array = cv2.imread(image_path)
    # hf.image_show(img_array)
    # # ipdb.set_trace()
    # [height_old, width_old, _] = img_array.shape
    # ipdb.set_trace()
    # img = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)
    hf.image_show(img)
    # ipdb.set_trace()
    img = img.astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        # Plot result
        depth_array = pred[0,:,:,0]
        ipdb.set_trace()
        depth_bigger = hf.image_scale(depth_array, new_size=[height_old, width_old])
        colors = img_array.reshape((np.prod(depth_bigger.shape), 3))
        # make it smaller
        # image_small = hf.image_scale(img_array, new_size=depth_array.shape)
        # colors = image_small.reshape((np.prod(depth_array.shape),3))
        # ipdb.set_trace()
        transform = hf.npz_load('/home/hao/MyCode/disparity_estimation/transform.npz', 'transform')
        KK = hf.npz_load('/home/hao/MyCode/disparity_estimation/k.npz', 'KK')
        ipdb.set_trace()
        K_d = KK
        k_d_scale = hf.intrinsic_cal_scale(K_d, img_array, depth_array)
        # points, points_matrix = hf.depthTo3D(depth_array, k_d_scale)
        points, points_matrix = hf.depthTo3D(depth_bigger, k_d_scale)
        # points, points_matrix = hf.depthTo3D(depth_bigger, k_d_scale)
        points = (transform[:3, :3].dot(points.T) + transform[:3, 3, None]).T
        ply_name = 'try.ply'
        hf.mesh_to_ply(ply_name, points, colors=colors)
        # ipdb.set_trace()
        hf.image_show_plt(depth_array)
        # hf.image_save('test.png', depth_array)
        hf.image_save('test.png', depth_bigger)
        print('done')
        # fig = plt.figure()
        # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        # ii = plt.imshow(asd, interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()
        return pred


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    # ipdb.set_trace()

    os._exit(0)


if __name__ == '__main__':
    main()
