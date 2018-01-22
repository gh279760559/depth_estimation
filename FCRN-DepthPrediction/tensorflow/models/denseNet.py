# @Author: Hao G <hao>
# @Date:   2018-01-22T15:05:42+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-22T15:07:30+00:00



from kaffe.tensorflow import Network

class DenseNet121UpProj(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv1_bn')
             .max_pool(3, 3, 2, 2, padding=None, name='pool1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_1_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_1_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_1_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_1_x2'))

        (self.feed('pool1',
                   'conv2_1_x2')
             .concat(3, name='concat_2_1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_2_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_2_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_2_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_2_x2'))

        (self.feed('concat_2_1',
                   'conv2_2_x2')
             .concat(3, name='concat_2_2')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_3_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_3_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_3_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_3_x2'))

        (self.feed('concat_2_2',
                   'conv2_3_x2')
             .concat(3, name='concat_2_3')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_4_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_4_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_4_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_4_x2'))

        (self.feed('concat_2_3',
                   'conv2_4_x2')
             .concat(3, name='concat_2_4')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_5_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_5_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_5_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_5_x2'))

        (self.feed('concat_2_4',
                   'conv2_5_x2')
             .concat(3, name='concat_2_5')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_6_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_6_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_6_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_6_x2'))

        (self.feed('concat_2_5',
                   'conv2_6_x2')
             .concat(3, name='concat_2_6')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv2_blk_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_blk')
             .avg_pool(2, 2, 2, 2, name='pool2')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_1_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_1_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_1_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_1_x2'))

        (self.feed('pool2',
                   'conv3_1_x2')
             .concat(3, name='concat_3_1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_2_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_2_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_2_x2'))

        (self.feed('concat_3_1',
                   'conv3_2_x2')
             .concat(3, name='concat_3_2')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_3_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_3_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_3_x2'))

        (self.feed('concat_3_2',
                   'conv3_3_x2')
             .concat(3, name='concat_3_3')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_4_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_4_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_4_x2'))

        (self.feed('concat_3_3',
                   'conv3_4_x2')
             .concat(3, name='concat_3_4')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_5_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_5_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_5_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_5_x2'))

        (self.feed('concat_3_4',
                   'conv3_5_x2')
             .concat(3, name='concat_3_5')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_6_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_6_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_6_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_6_x2'))

        (self.feed('concat_3_5',
                   'conv3_6_x2')
             .concat(3, name='concat_3_6')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_7_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_7_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_7_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_7_x2'))

        (self.feed('concat_3_6',
                   'conv3_7_x2')
             .concat(3, name='concat_3_7')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_8_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_8_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_8_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_8_x2'))

        (self.feed('concat_3_7',
                   'conv3_8_x2')
             .concat(3, name='concat_3_8')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_9_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_9_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_9_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_9_x2'))

        (self.feed('concat_3_8',
                   'conv3_9_x2')
             .concat(3, name='concat_3_9')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_10_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_10_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_10_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_10_x2'))

        (self.feed('concat_3_9',
                   'conv3_10_x2')
             .concat(3, name='concat_3_10')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_11_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_11_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_11_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_11_x2'))

        (self.feed('concat_3_10',
                   'conv3_11_x2')
             .concat(3, name='concat_3_11')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_12_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_12_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_12_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3_12_x2'))

        (self.feed('concat_3_11',
                   'conv3_12_x2')
             .concat(3, name='concat_3_12')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv3_blk_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_blk')
             .avg_pool(2, 2, 2, 2, name='pool3')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_1_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_1_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_1_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_1_x2'))

        (self.feed('pool3',
                   'conv4_1_x2')
             .concat(3, name='concat_4_1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_2_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_2_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_2_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_2_x2'))

        (self.feed('concat_4_1',
                   'conv4_2_x2')
             .concat(3, name='concat_4_2')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_3_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_3_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_3_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_3_x2'))

        (self.feed('concat_4_2',
                   'conv4_3_x2')
             .concat(3, name='concat_4_3')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_4_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_4_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_4_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_4_x2'))

        (self.feed('concat_4_3',
                   'conv4_4_x2')
             .concat(3, name='concat_4_4')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_5_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_5_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_5_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_5_x2'))

        (self.feed('concat_4_4',
                   'conv4_5_x2')
             .concat(3, name='concat_4_5')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_6_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_6_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_6_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_6_x2'))

        (self.feed('concat_4_5',
                   'conv4_6_x2')
             .concat(3, name='concat_4_6')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_7_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_7_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_7_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_7_x2'))

        (self.feed('concat_4_6',
                   'conv4_7_x2')
             .concat(3, name='concat_4_7')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_8_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_8_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_8_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_8_x2'))

        (self.feed('concat_4_7',
                   'conv4_8_x2')
             .concat(3, name='concat_4_8')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_9_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_9_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_9_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_9_x2'))

        (self.feed('concat_4_8',
                   'conv4_9_x2')
             .concat(3, name='concat_4_9')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_10_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_10_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_10_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_10_x2'))

        (self.feed('concat_4_9',
                   'conv4_10_x2')
             .concat(3, name='concat_4_10')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_11_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_11_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_11_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_11_x2'))

        (self.feed('concat_4_10',
                   'conv4_11_x2')
             .concat(3, name='concat_4_11')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_12_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_12_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_12_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_12_x2'))

        (self.feed('concat_4_11',
                   'conv4_12_x2')
             .concat(3, name='concat_4_12')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_13_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_13_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_13_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_13_x2'))

        (self.feed('concat_4_12',
                   'conv4_13_x2')
             .concat(3, name='concat_4_13')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_14_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_14_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_14_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_14_x2'))

        (self.feed('concat_4_13',
                   'conv4_14_x2')
             .concat(3, name='concat_4_14')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_15_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_15_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_15_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_15_x2'))

        (self.feed('concat_4_14',
                   'conv4_15_x2')
             .concat(3, name='concat_4_15')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_16_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_16_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_16_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_16_x2'))

        (self.feed('concat_4_15',
                   'conv4_16_x2')
             .concat(3, name='concat_4_16')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_17_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_17_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_17_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_17_x2'))

        (self.feed('concat_4_16',
                   'conv4_17_x2')
             .concat(3, name='concat_4_17')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_18_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_18_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_18_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_18_x2'))

        (self.feed('concat_4_17',
                   'conv4_18_x2')
             .concat(3, name='concat_4_18')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_19_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_19_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_19_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_19_x2'))

        (self.feed('concat_4_18',
                   'conv4_19_x2')
             .concat(3, name='concat_4_19')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_20_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_20_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_20_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_20_x2'))

        (self.feed('concat_4_19',
                   'conv4_20_x2')
             .concat(3, name='concat_4_20')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_21_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_21_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_21_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_21_x2'))

        (self.feed('concat_4_20',
                   'conv4_21_x2')
             .concat(3, name='concat_4_21')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_22_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_22_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_22_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_22_x2'))

        (self.feed('concat_4_21',
                   'conv4_22_x2')
             .concat(3, name='concat_4_22')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_23_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_23_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_23_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_23_x2'))

        (self.feed('concat_4_22',
                   'conv4_23_x2')
             .concat(3, name='concat_4_23')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_24_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_24_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_24_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv4_24_x2'))

        (self.feed('concat_4_23',
                   'conv4_24_x2')
             .concat(3, name='concat_4_24')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv4_blk_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_blk')
             .avg_pool(2, 2, 2, 2, name='pool4')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_1_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_1_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_1_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_1_x2'))

        (self.feed('pool4',
                   'conv5_1_x2')
             .concat(3, name='concat_5_1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_2_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_2_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_2_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_2_x2'))

        (self.feed('concat_5_1',
                   'conv5_2_x2')
             .concat(3, name='concat_5_2')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_3_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_3_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_3_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_3_x2'))

        (self.feed('concat_5_2',
                   'conv5_3_x2')
             .concat(3, name='concat_5_3')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_4_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_4_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_4_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_4_x2'))

        (self.feed('concat_5_3',
                   'conv5_4_x2')
             .concat(3, name='concat_5_4')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_5_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_5_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_5_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_5_x2'))

        (self.feed('concat_5_4',
                   'conv5_5_x2')
             .concat(3, name='concat_5_5')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_6_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_6_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_6_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_6_x2'))

        (self.feed('concat_5_5',
                   'conv5_6_x2')
             .concat(3, name='concat_5_6')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_7_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_7_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_7_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_7_x2'))

        (self.feed('concat_5_6',
                   'conv5_7_x2')
             .concat(3, name='concat_5_7')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_8_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_8_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_8_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_8_x2'))

        (self.feed('concat_5_7',
                   'conv5_8_x2')
             .concat(3, name='concat_5_8')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_9_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_9_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_9_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_9_x2'))

        (self.feed('concat_5_8',
                   'conv5_9_x2')
             .concat(3, name='concat_5_9')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_10_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_10_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_10_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_10_x2'))

        (self.feed('concat_5_9',
                   'conv5_10_x2')
             .concat(3, name='concat_5_10')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_11_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_11_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_11_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_11_x2'))

        (self.feed('concat_5_10',
                   'conv5_11_x2')
             .concat(3, name='concat_5_11')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_12_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_12_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_12_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_12_x2'))

        (self.feed('concat_5_11',
                   'conv5_12_x2')
             .concat(3, name='concat_5_12')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_13_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_13_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_13_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_13_x2'))

        (self.feed('concat_5_12',
                   'conv5_13_x2')
             .concat(3, name='concat_5_13')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_14_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_14_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_14_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_14_x2'))

        (self.feed('concat_5_13',
                   'conv5_14_x2')
             .concat(3, name='concat_5_14')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_15_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_15_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_15_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_15_x2'))

        (self.feed('concat_5_14',
                   'conv5_15_x2')
             .concat(3, name='concat_5_15')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_16_x1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv5_16_x1')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_16_x2_bn')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv5_16_x2'))

        (self.feed('concat_5_15',
                   'conv5_16_x2')
             .concat(3, name='concat_5_16')
             .batch_normalization({'scale_offset': False}, relu=True, name='conv5_blk_bn')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='layer1')
             .batch_normalization(relu=False, name='layer1_BN')
             .up_project([3, 3, 1024, 512], id = '2x', stride = 1, BN=True)
             .up_project([3, 3, 512, 256], id = '4x', stride = 1, BN=True)
             .up_project([3, 3, 256, 128], id = '8x', stride = 1, BN=True)
             .up_project([3, 3, 128, 64], id = '16x', stride = 1, BN=True)
             .dropout(name = 'drop', keep_prob = 1.)
             .conv(3, 3, 1, 1, 1, name = 'ConvPred'))
            #  .avg_pool(8, 8, 1, 1, padding='VALID', name='pool5')
            #  .conv(1, 1, 1000, 1, 1, relu=False, name='fc6'))
