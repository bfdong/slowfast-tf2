import os
import tensorflow as tf
import random
import numpy as np
import time
import threading


def map_function(sample_label,root_dir,num_frames_per_clip,sample_rate,width,height,crop_size,position=-1):
    ss = tf.strings.split(tf.strings.strip(sample_label))
    rgb_ret = []
    index = tf.random.uniform([1], minval=1, maxval=tf.strings.to_number(ss[1], out_type=tf.dtypes.int32)-num_frames_per_clip, dtype=tf.int32)[0]
    for i in range(0,num_frames_per_clip,sample_rate):
        image_name = root_dir+"/"+ss[0]+"/"+ tf.strings.as_string(index+i,width=5,fill='0')+".jpg"
        img_raw = tf.io.read_file(image_name)
        image = tf.image.decode_jpeg(img_raw)
        image = tf.image.resize(image,(height,width))
        if position == -1:
            image = tf.image.random_crop(image,[crop_size,crop_size,3])
        else:
            offet_h = int((height - crop_size) / 2)
        offet_w = int((width - crop_size) / 2)
        image = tf.image.crop_to_bounding_box(image, offet_h, offet_w, crop_size, crop_size)
        image = tf.cast(image,tf.float32)
        image = image - 128.0
        image = tf.expand_dims(image,0)
        rgb_ret.append(image)
    return tf.concat(rgb_ret,0),tf.strings.to_number(ss[2],out_type=tf.dtypes.int32)