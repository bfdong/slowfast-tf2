import os
import sys
sys.path.append('./')
import time
import numpy
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import slowfast
import tf_util



import math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import namedtuple
np.set_printoptions(suppress=True)
flags = tf.compat.v1.app.flags
gpu_num = 1

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 2, 'Number of steps to run epoch.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('sample_rate', 1, 'Sample rate for clib')
flags.DEFINE_integer('width', 172, 'Crop_size')
flags.DEFINE_integer('height', 128, 'Crop_size')
flags.DEFINE_integer('crop_size', 112, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 14, 'The num of class')
flags.DEFINE_integer('block_num', 0, 'The num of nonlocal block')
flags.DEFINE_bool('use_nonlocal', True, 'use or not nonlocal')
flags.DEFINE_float('weight_decay', 0.001, 'weight decay')

FLAGS = flags.FLAGS
HParams = namedtuple('HParams',['batch_size', 'num_classes', 'use_bottleneck', 'weight_decay_rate', 'relu_leakiness'])
hps = HParams(FLAGS.batch_size, FLAGS.classics, True, FLAGS.weight_decay, 0)

root_dir = "/home/vdtap/data"
save_dir = "./checkpoint/slowfast"
'''
video_name  total_frame  label
v_Basketball_g08_c01 451 1
v_Basketball_g08_c02 341 1
'''
test_file = '/home/vdtap/label/test.txt'





def run_testing():
    NET = slowfast.SLOWFAST(hps, 'test', 'no_nonlocal')
    model = NET.build_model(

    input_shape=(FLAGS.num_frame_per_clib, FLAGS.crop_size, FLAGS.crop_size, FLAGS.rgb_channels))

    # model = SlowFast_Network(clip_shape=[FLAGS.num_frame_per_clib,FLAGS.crop_size,FLAGS.crop_size,3],num_class=FLAGS.classics,alpha=8,beta=1/8,tau=8,method='T_conv')

    ckpt = tf.train.get_checkpoint_state(save_dir)
    print(ckpt)

    if ckpt and ckpt.model_checkpoint_path:
        checkpoint = tf.train.Checkpoint(myModel=model)
        checkpoint.restore(ckpt.model_checkpoint_path)
    else:
        print("load model error")

    file = open(test_file, 'r')
    lines = list(file)
    id_list = list(range(0, len(lines)))

    t = tf.data.Dataset.from_tensor_slices(lines)

    batch_t = t.map(lambda x: input_data.map_function(x, root_dir, FLAGS.num_frame_per_clib, FLAGS.sample_rate, FLAGS.width, FLAGS.height, FLAGS.crop_size, 1))

    batch_t = batch_t.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    step = 0
    predictions = []
    all_label = []

    for data_x, label_y in batch_t:
        y_pred, logits = model.predict(data_x)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label_y, logits))
        predictions.extend(y_pred)
        all_label.extend(label_y)
        step += 1
        print("step:", step, loss, label_y, y_pred[0], logits[0])

        tf_util.topk(predictions, all_label, id_list)
    y_pred = tf.argmax(predictions, 1)
    conf = tf.math.confusion_matrix(all_label, y_pred, num_classes=FLAGS.classics).numpy() # 计算混淆矩阵

    print(conf)
    print(conf / conf.sum(axis=1)[:, np.newaxis])

    np.savetxt('a.csv', conf / conf.sum(axis=1)[:, np.newaxis], delimiter=",", fmt='%f')
    print(ckpt)
    print('done!')

    return 0



def main(_):

    run_testing()



if __name__ == '__main__':

    assert tf.executing_eagerly();

    run_testing();