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


class GCN():
    def __init__(self,batch_norm_decay=0.99,batch_norm_epsilon=1e-3,):

        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        # 
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self.num_class = 4
        self.median_feature = 16
        self.filters = [64, 256, 512, 1024, 2048]
        self.strides = [2, 2, 1, 1]
        self.n = [3, 4, 6, 3]

    def forward_pass(self, x):
        """Build the core model within the graph"""
        with tf.variable_scope('resnet_v2_50', reuse=tf.AUTO_REUSE):
            size = tf.shape(x)[1:3]
            res = []
            x = x - [_R_MEAN, _G_MEAN, _B_MEAN]

            x = self._conv(x, 7, 64, 2, 'conv1', False, False)
            #x = self._max_pool(x, 3, 2, 'max')
            
            res_func = self._bottleneck_residual_v2
            gcn_func = self._gcn_block
            br_func  = self._br_block
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
        print(len(res))
        with tf.variable_scope('gcn', reuse=tf.AUTO_REUSE):
            #x = self._atrous_spatial_pyramid_pooling(x)
            #x = self._conv(x, 1, 5, 1, 'logits', False, False)
            #x = tf.image.resize_bilinear(x, size)
            gcn5 = self._gcn_block(res[3], self.median_feature, 3, 'gcn5')
            br5  = self._br_block(gcn5, self.median_feature, 'br5')
            br5_size  = br5.get_shape().as_list()
           
            #upbr5  = tf.image.resize_bilinear(br5, (br5_size[1]*self.strides[3],br5_size[2]*self.strides[3]))
            #upbr5  = self._conv(upbr5, 1, self.median_feature, strides=1, scope='br_conv', batch_norm=False)
            
          
            br4 = self._deconv(res[2],br5,'deconv4')
            br4_size  = br4.get_shape().as_list()

            upbr4  = tf.image.resize_bilinear(br4, (br4_size[1]*self.strides[2],br4_size[2]*self.strides[2]))
            upbr4  = self._conv(upbr4, 1, self.median_feature, strides=1, scope='br4_conv', batch_norm=False)
            
            br3 = self._deconv(res[1], upbr4, 'deconv1')
            br3_size  = br3.get_shape().as_list()
            upbr3  = tf.image.resize_bilinear(br3, (br3_size[1]*self.strides[1],br3_size[2]*self.strides[1]))
            upbr3  = self._conv(upbr3, 1, self.median_feature, strides=1, scope='br3_conv', batch_norm=False)

            br2 = self._deconv(res[0], upbr3, 'deconv2')
            br2_size  = br2.get_shape().as_list()
            upbr2  = tf.image.resize_bilinear(br2, (br2_size[1]*self.strides[0],br2_size[2]*self.strides[0]))
            upbr2  = self._conv(upbr2, 1, self.median_feature, strides=1, scope='br2_conv', batch_norm=False)
           
            
            br  = self._br_block(upbr2, self.median_feature, 'br_1')
            br_size  = br.get_shape().as_list()
            upbr  = tf.image.resize_bilinear(br, (br_size[1]*2,br_size[2]*2))
            upbr  = self._conv(upbr, 1, self.median_feature, strides=1, scope='br_conv', batch_norm=False)
            upbr  = self._br_block(upbr, self.median_feature, 'br_2')
            output = self._conv(upbr, 1, self.num_class, 1, 'output', False, False)
            return output

    def _A_ASPP(self):
        pass
        
    
    def _deconv(self,low,high, scope):
        with tf.variable_scope(scope):
            gcn = self._gcn_block(low, self.median_feature, 3, 'gcn')
            br  = self._br_block(gcn, self.median_feature, 'br1')
            print(high.get_shape(),low.get_shape(),gcn.get_shape(),br.get_shape())
            br  = tf.add(high, br)
            br  = self._br_block(br, self.median_feature, 'br2')
            return br
        
    def _br_block(self,x, filters,scope='br'):
        x_shape = x.get_shape().as_list()
        with tf.variable_scope(scope):       
            br = self._conv(x, 3, filters, 1, 'conv1', False, True)
            br = self._conv(br, 3, filters, 1, 'conv2', False, False)

            if x_shape[-1] != filters:
                short_cut = self._conv(x, 1, filters, 1, 'shortcut', False, False)
            else:
                short_cut = x
            x = tf.add(br, short_cut)
            return x
    
    def _gcn_block(self,x, filters,kernel_size=3, scope='gcn'):

        with tf.variable_scope(scope):
            x1 = self._conv_second(x, [1,kernel_size], filters, [1,1], 'x1_1')
            x1 = self._conv_second(x1, [kernel_size,1], filters, [1,1], 'x1_2')
            
            
            x2 = self._conv_second(x, [kernel_size,1], filters, [1,1], 'x2_1')
            x2 = self._conv_second(x2, [1,kernel_size], filters, [1,1], 'x2_2')
            
            out = tf.add(x1, x2)
            return out
            
   
   
    def _conv_second(self,x, kernel_size, filters, strides, scope, activation=False):
        """Convolution."""
        with tf.variable_scope(scope):
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name='weights',shape=[kernel_size[0], kernel_size[1], x_shape[3], filters])
            x = tf.nn.conv2d(input=x,filter=w,padding='SAME',strides=[1, strides[0], strides[1], 1],name='conv', )
            b = tf.get_variable(name='biases', shape=[filters])
            x = x + b
            #if activation:
            #    x = tf.nn.relu(x)
            return x
            
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

    def _bottleneck_residual_v2(self,x,in_filter,out_filter,stride):

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

    def _conv(self,x, kernel_size, filters, strides, scope, batch_norm=False, activation=False, rate=None):
        """Convolution."""
        with tf.variable_scope(scope):
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name='weights',shape=[kernel_size, kernel_size, x_shape[3], filters])
            if rate == None:
                x = tf.nn.conv2d(input=x,filter=w,padding='SAME',strides=[1, strides, strides, 1],name='conv', )
            else:
                x = tf.nn.atrous_conv2d(value=x,filters=w, padding='SAME',name='conv', rate=rate)

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
