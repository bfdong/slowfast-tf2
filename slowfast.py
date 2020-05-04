# -- coding: UTF-8 --

from collections import namedtuple
import numpy as np
import tensorflow as tf





class SLOWFAST(object):

    """SLOWFAST model."""



    def __init__(self, hps,  mode):

        self.hps = hps
        self.mode = mode

    # build_model

    def build_model(self,input_shape):

        x_input = tf.keras.layers.Input(shape=input_shape,dtype=tf.float32, name='input_node')

        fast_x = x_input
        fast_x = tf.keras.layers.Conv3D(filters=8, kernel_size=[5, 7, 7], strides=(1, 2, 2), padding='SAME',
                                        name='fast_conv1')(fast_x)

        fast_x = tf.keras.layers.BatchNormalization()(fast_x)
        fast_x = tf.keras.layers.ReLU()(fast_x)
        fast_x = tf.keras.layers.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(fast_x)



        #slow_x = x_input[:, ::8, ...]
        slow_x = tf.gather(x_input, tf.range(0, input_shape[0], 8), axis=1)
        slow_x = tf.keras.layers.Conv3D(filters=64,kernel_size=[1, 7, 7],strides=(1, 2, 2),padding = 'SAME',
                                                        name='slow_conv1')(slow_x)
        slow_x = tf.keras.layers.BatchNormalization()(slow_x)
        slow_x = tf.keras.layers.ReLU()(slow_x)
        slow_x = tf.keras.layers.MaxPool3D(pool_size=(1, 3, 3), strides=(1,2,2))(slow_x)


        concat1 =  tf.keras.layers.Conv3D(filters=16, kernel_size=[5, 1, 1], strides=(8, 1, 1), padding='SAME', name='concat1')(fast_x)
        slow_x = tf.keras.layers.Concatenate(axis=-1)([slow_x, concat1])




        block_num = [3, 4, 6, 3]
        filters_slow_out = [256, 512, 1024, 2048]
        filters_fast_out = [32, 64, 128, 256]
        inflate_list_slow = [False, False, True, True]
        inflate_list_fast = [True, True, True, True]

        res_func = self._bottleneck_residual
        for index in range(0, 4):
            for i in range(0, block_num[index]):
                if i == 0:
                    if index != 0 :
                        slow_x = res_func(slow_x, filters_slow_out[index], [1, 2, 2],
                                          inflate=inflate_list_slow[index],need_short=True)
                    else:
                        slow_x = res_func(slow_x,  filters_slow_out[index],
                                          [1, 1, 1],  inflate=inflate_list_slow[index],need_short=True)

                else:
                    slow_x = res_func(slow_x, filters_slow_out[index], [1, 1, 1],
                                      inflate=inflate_list_slow[index])

                if i == 0:
                    if index != 0 :
                        fast_x = res_func(fast_x,  filters_fast_out[index], [1, 2, 2],
                                          inflate=inflate_list_fast[index],need_short=True)
                    else:
                        fast_x = res_func(fast_x, filters_fast_out[index],
						[1, 1, 1],  inflate=inflate_list_fast[index],need_short=True)

                else:
                    fast_x = res_func(fast_x, filters_fast_out[index], [1, 1, 1],
                                      inflate=inflate_list_fast[index])

            if index != 3:
                concat = tf.keras.layers.Conv3D(filters=filters_fast_out[index]*2, kernel_size=[5, 1, 1], strides=(8, 1, 1), padding='SAME')(fast_x)
                slow_x = tf.keras.layers.Concatenate(axis=-1)([slow_x, concat])

        slow_x = tf.keras.layers.GlobalAveragePooling3D()(slow_x)
        fast_x = tf.keras.layers.GlobalAveragePooling3D()(fast_x)
        global_pool= tf.keras.layers.Concatenate(axis=-1)([slow_x, fast_x])

        if self.mode == 'train':
            global_pool = tf.keras.layers.Dropout(0.5)(global_pool)
        logits = tf.keras.layers.Dense( units=self.hps.num_classes)(global_pool)
        predictions = tf.keras.layers.Softmax()(logits)
        model = tf.keras.Model(inputs=x_input, outputs=[predictions, logits])

        return model



    def _bottleneck_residual(self, x, out_filter, stride=[1, 1, 1], inflate=False,need_short = False):
        orig_x = x
        #a

        if inflate:
            length = 3
        else:
            length = 1

        x = tf.keras.layers.Conv3D(filters=int(out_filter/4), kernel_size=[length, 1, 1], strides=stride, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)



        #b
        x = tf.keras.layers.Conv3D(filters=int(out_filter/4), kernel_size=[1, 3, 3],  padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        #c
        x = tf.keras.layers.Conv3D(filters=out_filter, kernel_size=[1, 1, 1], padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        #shorcut
        if need_short :
            orig_x = tf.keras.layers.Conv3D(filters=out_filter, kernel_size=[1, 1, 1],strides=stride, padding='SAME')(orig_x)
            orig_x = tf.keras.layers.BatchNormalization()(orig_x)

        x += orig_x
        x = tf.keras.layers.ReLU()(x)


        return x