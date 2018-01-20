
from datetime import datetime
import os.path
import time
import math
import random

import numpy as np
import tensorflow as tf
import glob
from tensorflow import gfile

from models import Temporal_Stream
from inputs import Temporal_Input

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# command line options
FLAGS = tf.app.flags.FLAGS

# Data flags
tf.app.flags.DEFINE_string('data_dir', os.path.abspath('data'),
                           """Data directory""")
tf.app.flags.DEFINE_string('save_dir', os.path.abspath('runs'),
                           """Directory to save checkpoints and summaries""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_every', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('summary_every', 100,
                            """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_every', 1000,
                            """How often to checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 70000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_classes', 101,
                            """Number of classes (UCF-101: 101, HMDB-51: 51).""")

# Input flags
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('image_size', 24,
                            """Spatial size of input.""")
tf.app.flags.DEFINE_integer('temporal_depth', 160,
                            """Temporal depth of input.""")

# Training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('momentum_rate', 0.9,
                          """Momentum rate.""")

tf.app.flags.DEFINE_string('opt_type', 'Momentum',
                           """Optimizer type (Momentum or Adam).""")
tf.app.flags.DEFINE_integer('decay_step', 30e3,
                            """Number of steps between decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """Decay rate for exponential moving average""")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.2,
                          """Dropout keep probability.""")
tf.app.flags.DEFINE_float('wd_coeff', 0.005,
                          """L2 weight decay coefficient.""")


# summaries directory
train_summary_dir = os.path.join(FLAGS.save_dir, 'summaries', 'train')

# checkpoint directory
train_checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints')
if not os.path.exists(train_checkpoint_dir):
    os.makedirs(train_checkpoint_dir)


def _add_grad_summaries(grads_and_vars):
    '''
    Add gradient summaries
    :param grads_and_vars: list of (gradient, variable) pairs as returned by compute_gradients()
    :return: merged gradient and variable summaries
    '''
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    return grad_summaries_merged


def train(total_loss, var_list=None):
    '''
    Training function (currently restricted to single GPU training)
    :param total_loss: total loss (cross entropy + weight decay)
    :param var_list: variable list for training/gradient computation
    :return: training operation to run, training summaries (gradients, learning rate), training step
    '''
    with tf.name_scope('optimizer'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        if FLAGS.opt_type == 'Momentum':
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                  global_step,
                                  FLAGS.decay_step,
                                  FLAGS.learning_rate_decay_factor,
                                  staircase=True)

            lr_summary = tf.summary.scalar('learning_rate', lr)

            optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum = FLAGS.momentum_rate)
        elif FLAGS.opt_type == 'Adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)
            lr_summary = tf.summary.scalar('learning_rate', optimizer._lr)

        grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
        opt_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # add grad_summaries
    grad_summaries = _add_grad_summaries(grads_and_vars)

    train_summaries = tf.summary.merge([grad_summaries, lr_summary])

    return opt_op, train_summaries, global_step


def build_graph():
    '''
    Build model graph
    :return: saver object
    '''
    train_files = glob.glob(os.path.join(FLAGS.data_dir, '*.bin'))

    inputs = Temporal_Input(FLAGS.data_dir, FLAGS.batch_size, FLAGS.image_size, FLAGS.temporal_depth)
    train_data = inputs.generate_batches_to_train(train_files)

    train_model = Temporal_Stream(train_data, FLAGS.num_classes, FLAGS.wd_coeff, FLAGS.dropout_keep_prob)

    # initialize optimizer
    opt_op, train_summaries, global_step = train(train_model.total_loss)

    summary_op = tf.summary.merge([train_model.merged_summaries, train_summaries])

    # Print model breakdown
    model_breakdown(train_model)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([opt_op]):
        train_op = tf.group(variable_averages_op)

    
    # add ops to graph collections
    tf.add_to_collection('global_step', global_step)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('summary_op', summary_op)
    tf.add_to_collection('total_loss', train_model.total_loss)
    tf.add_to_collection('accuracy', train_model.accuracy)

    # create a saver.
    saver = tf.train.Saver(tf.global_variables())

    return saver


def get_meta_filename():
    '''
    Check if model already exists and get metagraph file to restore
    :return: metagraph file name (string)
    '''
    latest_checkpoint = tf.train.latest_checkpoint(train_checkpoint_dir)
    if not latest_checkpoint:
        print('No checkpoint file found. Starting new model')
        return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
        print('No meta graph file found. Building a new model')
        return None
    else:
        print('Found meta file at {}\n'.format(meta_filename))
        return meta_filename


def model_breakdown(model):
    '''
    Print breakdown in terms of number of activations and weights for model
    :param model: Model object
    :return: None
    '''
    print('Model cost breakdown')
    print('\nActivations:')
    #activations = tf.get_collection('activations')[0]
    for key, value in model.activations.iteritems():
        print('Layer %s: %i bytes' % (key, value))
    print('Total activation memory cost (~x2 for backprop to store gradients and activations): %i bytes' %
          sum(model.activations.values()))

    print('\nWeights:')
    #weights = tf.get_collection('weights')[0]
    for key, value in model.weights.iteritems():
        print('Layer %s: %i' % (key, value))
    print('Total number of weights: %i \n' % sum(model.weights.values()))


def run():
    '''
    Run model graph in session
    :return: None
    '''
    graph = tf.Graph()
    with graph.as_default():
        meta_filename = get_meta_filename()
        if meta_filename is not None:
            latest_checkpoint = os.path.splitext(meta_filename)[0]
            saver = tf.train.import_meta_graph(meta_filename, clear_devices=True)
        else:
            print('Training for %d steps' % FLAGS.max_steps)
            saver = build_graph()
           
        # Get ops from graph
        global_step = tf.get_collection('global_step')[0]
        train_op = tf.get_collection('train_op')[0]
        summary_op = tf.get_collection('summary_op')[0]
        total_loss = tf.get_collection('total_loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
    
    # Create session to run graph
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8)

    with tf.Session(config=session_conf, graph=graph) as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        
        if meta_filename is not None:
            saver.restore(sess, latest_checkpoint)
            
        # Writer for summaries
        summary_writer = tf.summary.FileWriter(train_summary_dir, graph)


        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # initial step 
        init_step = int(sess.run(global_step))+1

        try:
            while not coord.should_stop():
                for step in range(init_step,FLAGS.max_steps):
                    start_time = time.time()
                    _, train_loss, train_accuracy, _ = sess.run([train_op, total_loss, accuracy, global_step])

                    duration = time.time() - start_time

                    assert not np.isnan(train_loss), 'Model diverged with loss = NaN'

                    if step % FLAGS.log_every == 0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('%s: step %d, accuracy = %.2f%%, loss = %.2f, (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), step, train_accuracy, train_loss, examples_per_sec, sec_per_batch))

                    if step % FLAGS.summary_every == 0:
                        summary = sess.run(summary_op)
                        train_summary = tf.Summary()
                        train_summary.ParseFromString(summary)
                        train_summary.value.add(tag='accuracy', simple_value=train_accuracy.item())
                        summary_writer.add_summary(train_summary, step)

                    if step % FLAGS.checkpoint_every == 0 or (step+1) == FLAGS.max_steps:
                        path = saver.save(sess, os.path.join(train_checkpoint_dir, 'model.ckpt'), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
                        
                # When done, ask the threads to stop.
                coord.request_stop()

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    run()
