import os
import os
import sys



sys.path.append('../../')
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import slowfast

import math
import numpy as np
import multiprocessing as mt

np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from collections import namedtuple


flags = tf.compat.v1.app.flags
gpu_num = 1
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 10, 'Number of steps to run epoch.')
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
flags.DEFINE_float('weight_decay', 0.0005, 'weight decay')

FLAGS = flags.FLAGS



HParams = namedtuple('HParams',  ['batch_size', 'num_classes', 'use_bottleneck', 'weight_decay_rate', 'relu_leakiness'])
hps = HParams(FLAGS.batch_size, FLAGS.classics, True, FLAGS.weight_decay, 0)



root_dir = "/home/vdtap/data"
save_dir = "./checkpoint/slowfast"
'''
video_name  total_frame  label
v_Basketball_g08_c01 451 1
v_Basketball_g08_c02 341 1
'''
train_file = '/home/vdtap/label/train.txt'
pre_model_save_dir = './checkpoints/resnet_pretrain1'



def run_training():
    NET = slowfast.SLOWFAST(hps,  'train' )
    model = NET.build_model(input_shape=(FLAGS.num_frame_per_clib,FLAGS.crop_size,FLAGS.crop_size,FLAGS.rgb_channels))
    #model = SlowFast_Network(clip_shape=[FLAGS.num_frame_per_clib,FLAGS.crop_size,FLAGS.crop_size,3],num_class=FLAGS.classics,alpha=8,beta=1/8,tau=8,method='T_conv')

    model.summary(line_length=125,positions=[.40, .70, .80, 1.])
    ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)

    print("ckpt:",ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        print("load model",ckpt.model_checkpoint_path)
        checkpoint = tf.train.Checkpoint(myModel=model)
        checkpoint.restore(ckpt.model_checkpoint_path)

    file = open(train_file, 'r')
    lines = list(file)
    boundaries =[ x*int((len(lines)/FLAGS.batch_size)) for x in  [8]]
    print(boundaries)
    values = [0.0001,0.00001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9)
    t = tf.data.Dataset.from_tensor_slices(lines)
    batch_t = t.shuffle(len(lines)).repeat(FLAGS.epoch_steps)\
        .map(lambda x:  input_data.map_function(x,root_dir,FLAGS.num_frame_per_clib, FLAGS.sample_rate,FLAGS.width,FLAGS.height,FLAGS.crop_size),num_parallel_calls=8)

    batch_t = batch_t.batch(FLAGS.batch_size).prefetch(8)

    step = 0
    @tf.function
    def train_step(data_x,label_y):
        with tf.GradientTape() as tape:
            print("=============>>>",step)
            y_pred,logits = model(data_x)
            #print("logits",logits[0],y_pred[0])
            l2_loss = FLAGS.weight_decay*tf.reduce_sum([tf.nn.l2_loss(a) for a in model.trainable_variables])
            #all_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label_y,logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label_y,logits)) + l2_loss
            #print("loss:",l2_loss,loss-l2_loss,all_loss)
        grads = tape.gradient(loss, model.trainable_variables)  # 使用 model.variables 这一属性直接获得模型中的所有变量
        #print("grads",len(grads),grads[0].shape,grads[533].shape,grads[0][0][0][0][0],grads[532][0])

        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        return l2_loss,loss,logits,grads

    summary_writer = tf.summary.create_file_writer('./tensorboard')
    checkpoint = tf.train.Checkpoint(myModel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=save_dir, checkpoint_name='slowfast.ckpt', max_to_keep=5)

    for data,label in batch_t:
        start_time =time.time()
        l2_loss,loss,logits,grads = train_step(data,label)
        with summary_writer.as_default():                               # summary scalar
            tf.summary.scalar("loss", loss, step=step)
        duration = time.time() - start_time
        print("================",step,l2_loss, loss-l2_loss,label,duration)
        #for layer in model.layers:
        #    if layer.name == 'fast_conv1' or layer.name == 'dense':
        #        print("weight:",layer.weights[1])
        step += 1
        if step % 10000 == 0 :
            manager.save(checkpoint_number=step)


    manager.save(checkpoint_number=step)
    h5_file = save_dir+"/slowfast_model_{}.h5".format(step)
    print(h5_file)
    if os.path.exists(h5_file):
        os.remove(h5_file)
    #when save h5 format, it will be report failure ,the bug is fixed in 2.2

    #model.save(h5_file)



    return 0











def main(_):

    run_training()





if __name__ == '__main__':

    assert tf.executing_eagerly();

    run_training();