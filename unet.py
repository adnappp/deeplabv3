# coding:utf-8
"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import moving_averages

import tensorflow as tf


_R_MEAN = 119.23
_G_MEAN = 135.88
_B_MEAN = 138.98


class Unet():
    def __init__(self,
                 batch_norm_decay=0.99,
                 batch_norm_epsilon=1e-3,):

        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        # 
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self.num_class = 4
        self.filters = [64, 256, 512, 1024, 2048]
        self.strides = [2, 2, 2, 2]
        self.n = [3, 4, 6, 3]

    def forward_pass(self, x):
        """Build the core model within the graph"""
        with tf.variable_scope('resnet_v2_50', reuse=tf.AUTO_REUSE):
            size = tf.shape(x)[1:3]
            res = []
            x = x - [_R_MEAN, _G_MEAN, _B_MEAN]
            source = x
            x = self._conv(x, 7, 64, 2, 'conv1', False, False)
            #x = self._max_pool(x, 3, 2, 'max')
            res.append(x)
            res_func = self._bottleneck_residual_v2
      
            for i in range(4):
                with tf.variable_scope('block%d' % (i + 1)):
                    for j in range(self.n[i]):
                        with tf.variable_scope('unit_%d' % (j + 1)):
                            if j == 0:
                                x = res_func(x, self.filters[i], self.filters[i+1], 1)
                            elif j == self.n[i] - 1:
                                x = res_func(x, self.filters[i+1], self.filters[i+1], self.strides[i])
                            else:
                                x = res_func(x, self.filters[i+1], self.filters[i+1], 1)
                    res.append(x)
                tf.logging.info('the shape of features after block%d is %s' % (i+1, x.get_shape()))

        # 
        with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
            #x = self._atrous_spatial_pyramid_pooling(x)
            #x = self._conv(x, 1, 5, 1, 'logits', False, False)
            #x = tf.image.resize_bilinear(x, size)
            #b h//32 w//32 2048 
            x = self._conv(x, 1, self.filters[3], strides=1, scope='conv4_1', batch_norm=False, activation=True)
            x = self._conv(x, 3, self.filters[3], strides=1, scope='conv4_2', batch_norm=False, activation=True)
            x_size  = x.get_shape().as_list()
            x  = tf.image.resize_bilinear(x, (x_size[1]*2, x_size[2]*2))
            x = self._conv(x, 2, self.filters[3], strides=1, scope='conv4_3', batch_norm=False)
            
            x  = tf.concat([x,res[3]],-1)
            x = self._conv(x, 1, self.filters[3], strides=1, scope='conv3_1', batch_norm=False, activation=True)
            x = self._conv(x, 3, self.filters[3], strides=1, scope='conv3_2', batch_norm=False, activation=True)
            x  = tf.image.resize_bilinear(x, (x_size[1]*4, x_size[2]*4))
            x = self._conv(x, 2, self.filters[2], strides=1, scope='conv3_3', batch_norm=False)
            #print(x.get_shape(),res[2].get_shape())
            
            x  = tf.concat([x,res[2]],-1)
            x = self._conv(x, 1, self.filters[2], strides=1, scope='conv2_1', batch_norm=False, activation=True)
            x = self._conv(x, 3, self.filters[2], strides=1, scope='conv2_2', batch_norm=False, activation=True)
            x  = tf.image.resize_bilinear(x, (x_size[1]*8, x_size[2]*8))
            x = self._conv(x, 2, self.filters[1], strides=1, scope='conv2_3', batch_norm=False)
            #print(x.get_shape(),res[1].get_shape())
            
            x  = tf.concat([x,res[1]],-1)
            x = self._conv(x, 1, self.filters[1], strides=1, scope='conv1_1', batch_norm=False, activation=True)
            x = self._conv(x, 3, self.filters[1], strides=1, scope='conv1_2', batch_norm=False, activation=True)
            x  = tf.image.resize_bilinear(x, (x_size[1]*16, x_size[2]*16))
            x = self._conv(x, 2, self.filters[0], strides=1, scope='conv1_3', batch_norm=False)
            #print(x.get_shape(),res[0].get_shape())

            x = tf.concat([x, res[0]], -1)
            x = self._conv(x, 1, self.filters[0], strides=1, scope='conv_1', batch_norm=False, activation=True)
            x = self._conv(x, 3, self.filters[0], strides=1, scope='conv_2', batch_norm=False, activation=True)
            x = tf.image.resize_bilinear(x, (x_size[1] * 32, x_size[2] * 32))
            x = self._conv(x, 2, self.filters[0], strides=1, scope='conv_3', batch_norm=False)
            output = self._conv(x, 1, self.num_class, 1, 'output', False, False)

            return output
    def _A_ASPP(self):
        pass
        
    def _deconv(self,low,high, scope):
        with tf.variable_scope(scope):
            gcn = self._gcn_block(low, self.median_feature, 3, 'gcn')
            br  = self._br_block(gcn, self.median_feature, 'br1')
            #print(high.get_shape(),low.get_shape(),gcn.get_shape(),br.get_shape())
            br  = tf.add(high, br)
            br  = self._br_block(br, self.median_feature, 'br2')
            return br
    
    def _atrous_spatial_pyramid_pooling(self, x):
        """
        """
        with tf.variable_scope('ASSP_layers'):

            feature_map_size = tf.shape(x)

            image_level_features = tf.reduce_mean(x, [1, 2], keep_dims=True)
            image_level_features = self._conv(image_level_features, 1, 256, 1, 'global_avg_pool', True)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1],
                                                                                   feature_map_size[2]))

            at_pool1x1   = self._conv(x, kernel_size=1, filters=256, strides=1, scope='assp1', batch_norm=True)
            at_pool3x3_1 = self._conv(x, kernel_size=3, filters=256, strides=1, scope='assp2', batch_norm=True, rate=6)
            at_pool3x3_2 = self._conv(x, kernel_size=3, filters=256, strides=1, scope='assp3', batch_norm=True, rate=12)
            at_pool3x3_3 = self._conv(x, kernel_size=3, filters=256, strides=1, scope='assp4', batch_norm=True, rate=18)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3)

            net = self._conv(net, kernel_size=1, filters=256, strides=1, scope='concat', batch_norm=True)

            return net

    def _bottleneck_residual_v2(self,
                                x,
                                in_filter,
                                out_filter,
                                stride,):

        """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

        with tf.variable_scope('bottleneck_v2'):
            origin_x = x
            with tf.variable_scope('preact'):
                preact = self._batch_norm(x)
                preact = self._relu(preact)

            residual = self._conv(preact, 1, out_filter // 4, stride, 'conv1', True, True)
            residual = self._conv(residual, 3, out_filter // 4, 1, 'conv2', True, True)
            residual = self._conv(residual, 1, out_filter, 1, 'conv3', False, False)

            if in_filter != out_filter:
                short_cut = self._conv(preact, 1, out_filter, stride, 'shortcut', False, False)
            else:
                short_cut = self._subsample(origin_x, stride, 'shortcut')
            x = tf.add(residual, short_cut)
            return x

    def _conv(self,
              x,
              kernel_size,
              filters,
              strides,
              scope,
              batch_norm=False,
              activation=False,
              rate=None
              ):
        """Convolution."""
        with tf.variable_scope(scope):
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name='weights',
                                shape=[kernel_size, kernel_size, x_shape[3], filters])
            if rate == None:
                x = tf.nn.conv2d(input=x,
                                 filter=w,
                                 padding='SAME',
                                 strides=[1, strides, strides, 1],
                                 name='conv', )
            else:
                x = tf.nn.atrous_conv2d(value=x,
                                        filters=w,
                                        padding='SAME',
                                        name='conv',
                                        rate=rate)

            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    x = self._batch_norm(x)
            else:
                b = tf.get_variable(name='biases', shape=[filters])
                x = x + b
            if activation:
                x = tf.nn.relu(x)
            return x

    def _batch_norm(self, x):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))
        beta = tf.get_variable(name='beta',
                               shape=params_shape,
                               initializer=tf.zeros_initializer)

        gamma = tf.get_variable(name='gamma',
                                shape=params_shape,
                                initializer=tf.ones_initializer)

        moving_mean = tf.get_variable(name='moving_mean',
                                      shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)

        moving_variance = tf.get_variable(name='moving_variance',
                                          shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        tf.add_to_collection('BN_MEAN_VARIANCE', moving_mean)
        tf.add_to_collection('BN_MEAN_VARIANCE', moving_variance)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean,
                                                                   self._batch_norm_decay,
                                                                   name='MovingAvgMean')
        update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                       variance,
                                                                       self._batch_norm_decay,
                                                                       name='MovingAvgVariance')

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        mean, variance = tf.cond(
            pred=self._is_training,
            true_fn=lambda: (mean, variance),
            false_fn=lambda: (moving_mean, moving_variance)
        )
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self._batch_norm_epsilon)
        return x

    def _relu(self, x):
        return tf.nn.relu(x)

    def _max_pool(self, x, pool_size, stride, scope):
        with tf.name_scope('max_pool') as name_scope:
            x = tf.layers.max_pooling2d(
                x, pool_size, stride, 'SAME', name=scope
            )
        return x

    def _avg_pool(self, x, pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size, stride, 'SAME')
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _global_avg_pool(self, x):
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4

            x = tf.reduce_mean(x, [1, 2])
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _concat(self, x, y):
        with tf.name_scope('concat') as name_scope:
            assert x.get_shape().ndims == 4
            assert y.get_shape().ndims == 4

            x = tf.concat([x, y], 3)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _subsample(self, inputs, stride, scope=None):
        """Subsamples the input along the spatial dimensions."""
        if stride == 1:
            return inputs
        else:
            return self._max_pool(inputs, 3, stride, scope)
