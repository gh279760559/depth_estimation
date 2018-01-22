# @Author: Hao G <hao>
# @Date:   2018-01-12T12:28:46+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-22T16:18:06+00:00


import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import ipdb
import handy_function as hf
import cv2
import pickle

import models


def method_FCRN(img, model_data_path):
    """FCRN method.

    img: img_array
    """
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    img = np.expand_dims(np.asarray(img), axis=0)
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    # net = models.DenseNet121UpProj({'data': input_node}, batch_size, 1, True)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        return pred[0, :, :, 0]


def method_CMCA(img, address=None):
    import requests
    SPARTAN_IP = '192.168.1.251'
    if address is None:
        address = SPARTAN_IP
    response = requests.post(
        'http://{}:5000/'.format(address),
        data=pickle.dumps(img, protocol=0),
        files={}
    )
    # res_json = response.json()
    # depth = pickle.loads(res_json['depth'].decode())
    if response.status_code != 200:
        raise Exception(
            "Network request failed {}: {}\n"
            "".format(response.status_code, response.content)
        )
    data = pickle.loads(response.content, encoding='bytes')
    depth = np.squeeze(data[b'depth'])
    return depth


def image_depth_measurement(depth_array, img_array):
    if(len(np.asarray(img_array).shape) == 2):
        [height_old, width_old] = np.asarray(img_array).shape
    else:
        [height_old, width_old, _] = np.asarray(img_array).shape
    # depth = hf.image_scale(depth_array, new_size=[height_old, width_old])
    # ipdb.set_trace()
    # colors = img_array.reshape((np.prod(depth.shape), 3))
    # make it smaller
    image_small = hf.image_scale(np.array(img_array), new_size=depth_array.shape)
    colors = image_small.reshape((np.prod(depth_array.shape), 3))
    depth = depth_array
    # ipdb.set_trace()
    return depth, colors


def mapTo3D(ply_name, depth, colors, k_d_scale, transform):
    points, points_matrix = hf.depthTo3D(depth, k_d_scale)
    # points, points_matrix = hf.depthTo3D(depth_bigger, k_d_scale)
    points = (transform[:3, :3].dot(points.T) + transform[:3, 3, None]).T
    # ipdb.set_trace()
    hf.mesh_to_ply(ply_name, points, colors=colors)


def predict(model_data_path, KK, transform, image_path, flag, file_name, if_portrait):

    # Read image
    # ipdb.set_trace()
    img_array = Image.open(image_path)
    # img_array = cv2.imread(image_path)
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    if(if_portrait):
        img_array = img_array.rotate(-90, expand=True)
    img_array = np.array(img_array)

    # img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    # img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    # ipdb.set_trace()
    # img_array = img_array.rotate(-90, expand=True)
    if(flag == 'FCRN'):
        img = cv2.resize(img_array, (304, 228), interpolation=cv2.INTER_CUBIC)
        # img = np.asarray(img_array.resize([304, 228], Image.BICUBIC))  # Image.ANTIALIAS
        depth_name = file_name + 'FCRN.png'
        ply_name = file_name + 'FCRN.ply'
    elif(flag == 'CMCA'):
        img = cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_CUBIC)
        # img = np.asarray(img_array.resize([320, 240], Image.BICUBIC))  # Image.BICUBIC
        depth_name = file_name + 'CMCA.png'
        ply_name = file_name + 'CMCA.ply'
    else:
        ipdb.set_trace()
        img = img_array[:int((img_array.shape[0] - 1) / 2), :, :]
        depth_array = img_array[int((img_array.shape[0] - 1) / 2):-1, :, 0]
        img_array = img
        depth_name = file_name + 'perception.png'
        ply_name = file_name + 'perception.ply'

    # # img_array = cv2.imread(image_path)
    # hf.image_show(img_array)
    # # ipdb.set_trace()
    # [height_old, width_old, _] = img_array.shape
    # ipdb.set_trace()
    # img = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)
    hf.image_show(img)
    # ipdb.set_trace()
    img = img.astype('float32')
    # ipdb.set_trace()
    if(flag == 'FCRN'):
        pred = method_FCRN(img, model_data_path)
        depth_map, colors = image_depth_measurement(pred, np.array(img_array))
    elif(flag == 'CMCA'):
        pred = method_CMCA(img)
        depth_map, colors = image_depth_measurement(pred, np.array(img_array))
    else:
        pred = depth_array
        depth_map = depth_array
        colors = img_array.reshape((np.prod(depth_array.shape), 3))
        ipdb.set_trace()
    hf.image_show(pred)

    hf.image_show(depth_map)

    # hf.image_save('test.png', depth_array)
    hf.image_save(depth_name, depth_map)
    # ipdb.set_trace()

    K_d = KK
    k_d_scale = hf.intrinsic_cal_scale(K_d, np.array(img_array), depth_map)  # depth_FCRN or CMCA does not matter

    mapTo3D(ply_name, depth_map, colors, k_d_scale, transform)
    # ipdb.set_trace()

    print('done')
    # fig = plt.figure()
    # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
    # ii = plt.imshow(asd, interpolation='nearest')
    # fig.colorbar(ii)
    # plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()
    # 7989.147220208-1.000, 7979.64259175-1.000, 7984.253418541-1.000
    file_name = '7984.253418541-1.000_downsampling'
    if_portrait = True
    saved_path = '/media/hao/DATA/MyCode/depth_estimation/FCRN-DepthPrediction/tensorflow/'
    transform = hf.npz_load(hf.path_join(saved_path, file_name + '-transform.npz'), 'transform')
    KK = hf.npz_load(hf.path_join(saved_path, file_name + '-k.npz'), 'KK')
    # Predict the image
    predict(args.model_path, KK, transform, args.image_paths, 'FCRN', file_name, if_portrait)
    # ipdb.set_trace()
    predict(args.model_path, KK, transform, args.image_paths, 'CMCA', file_name, if_portrait)

    # th test_on_one_image.lua -prev_model_file /media/hao/DATA/MyCode/relative_depth/src/results/hourglass3/NYU_795_800_c9_1e-3/Best_model_period1.t7 -input_image /media/hao/DATA/Arkit/test1/8091.827563333-1.000.jpeg -output_image asd.jpg -output_image1 asd1.jpg
    # predict(args.model_path, KK, transform, args.image_paths, 'new_one', file_name, if_portrait)
    # ipdb.set_trace()

    os._exit(0)


if __name__ == '__main__':
    main()
