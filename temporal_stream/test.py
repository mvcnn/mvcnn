from datetime import datetime
import os
import time
import math
import sys
import cPickle
import glob

import numpy as np
from six.moves import xrange
import tensorflow as tf

from models import Temporal_Stream
from inputs import Temporal_Input

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# command line options
FLAGS = tf.app.flags.FLAGS

# Data flags
tf.app.flags.DEFINE_string('data_dir', os.path.abspath('data'),
                           """Data directory""")
tf.app.flags.DEFINE_string('save_dir', os.path.abspath('runs'),
                           """Directory where checkpoints and summaries are saved""")
tf.app.flags.DEFINE_integer('num_classes', 101,
                            """Number of classes (UCF-101: 101, HMDB-51: 51).""")

# Input flags
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('image_size', 24,
                            """Spatial size of input.""")
tf.app.flags.DEFINE_integer('temporal_depth', 160,
                            """Temporal depth of input.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """Decay rate for exponential moving average""")

# Test flags
tf.app.flags.DEFINE_integer('num_test_crops_per_sample', 6,
                            """Number of test crops per sample to evaluate on.""")
tf.app.flags.DEFINE_integer('num_test_samples_per_file', 2,
                            """Number of test samples per file to evaluate on.""")



# summaries directory
test_summary_dir = os.path.join(FLAGS.save_dir, 'summaries', 'test')


# checkpoint directory
test_checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints')
checkpoint_prefix = os.path.join(test_checkpoint_dir, "model.ckpt")
if not os.path.exists(test_checkpoint_dir):
    os.makedirs(test_checkpoint_dir) 

    
def prediction(score, label):
    '''
    Generate prediction for video
    :param score: list of video scores
    :param label: video label
    :return: boolean (True for match), class index for maximum score
    '''
    label = int(label)
    correct_prediction = [True] if np.argmax(score) == label else [False]
    return correct_prediction, np.argmax(score)


def build_graph():
    '''
    Build model graph
    :return: saver object
    '''
    test_files = glob.glob(os.path.join(FLAGS.data_dir, '*.bin'))

    inputs = Temporal_Input(FLAGS.data_dir, FLAGS.batch_size, FLAGS.image_size, FLAGS.temporal_depth,
                           FLAGS.num_test_crops_per_sample, FLAGS.num_test_samples_per_file)
    test_data = inputs.generate_batches_to_eval(test_files, test = True)

    test_model = Temporal_Stream(test_data, FLAGS.num_classes, reuse=False, train=False)

    #remove contents of test_summary_dir
    contents = ('%s/.*' % test_summary_dir)
    sums = glob.glob(contents)
    for f in sums:
        os.remove(f)
    
    summary_op = tf.summary.merge([test_model.merged_summaries])
    '''
    # add ops to graph collections
    tf.add_to_collection('scores', test_model.scores)
    tf.add_to_collection('labels', test_data.labels)
    tf.add_to_collection('num_files', len(test_files))
    tf.add_to_collection('summary_op', summary_op)
    '''
    '''
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    '''
    saver = tf.train.Saver()
    
    return saver, test_model.scores, test_data.labels, len(test_files), summary_op
    

def run():
    '''
    Run model graph in session
    :return:None
    '''
    graph = tf.Graph()
    with graph.as_default():
        saver,scores,labels,num_files,summary_op = build_graph()
        '''
        scores = tf.get_collection('scores')[0]
        labels = tf.get_collection('labels')[0]
        num_files = tf.get_collection('num_files')[0]
        summary_op = tf.get_collection('summary_op')[0]
        '''

    # Create session to run graph
    session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)

    with tf.Session(graph=graph, config=session_conf) as sess:

        ckpt = tf.train.get_checkpoint_state(test_checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Found checkpoint file at {}\n'.format(ckpt.model_checkpoint_path))
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # extract global_step
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            raise Exception('No checkpoint file found')
            
        # Writer for summaries
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, graph)


        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                def get_scores(num_iter, sess=sess):
                    test_step = 0
                    score_array = np.empty((0,FLAGS.num_classes), float)
                    label_vec = []
                    start_time = time.time()
                    while test_step < num_iter:
                        # Add code for validation set here
                        scores_per_batch, labels_per_batch = sess.run([scores, labels]) 
                        score_array = np.append(score_array, scores_per_batch, axis=0)
                        label_vec.extend(labels_per_batch)
                        #print('Iteration %i' % test_step)
                        duration = time.time() - start_time
                        vids_processed = round(test_step*FLAGS.batch_size / (FLAGS.num_test_crops_per_sample * 
                                                                             FLAGS.num_test_samples_per_file))
                        print ('Videos processed: %i/%i (%.3f sec)' % (vids_processed, num_files, duration))
                        test_step += 1
                    return score_array, label_vec
                
                
                
                #I-frames
                # number of iterations = (number of test videos * number of crops) / batch size
                #P-frames
                num_test_crops_per_file = FLAGS.num_test_crops_per_sample*FLAGS.num_test_samples_per_file
                num_iter = math.ceil(float(num_files*num_test_crops_per_file)  / FLAGS.batch_size)
                score_array, label_vec = get_scores(num_iter)
                
                print score_array.shape, len(label_vec)
                
                cumulative_count = 0
                video_predictions = []
                for n in range(0,num_files):
                    count = num_test_crops_per_file
                    if cumulative_count + count < len(label_vec):
                        video_label = label_vec[cumulative_count:cumulative_count+count]
                        average_score = np.amax(score_array[cumulative_count:cumulative_count+count,:], axis=0)
                        assert all(x==video_label[0] for x in video_label)
                        is_correct, pred = prediction(average_score, video_label[0]) 
                        video_predictions.extend(is_correct)
                        cumulative_count += count

                        
                print len(video_predictions)
                    
                test_accuracy = float(np.sum(video_predictions))/len(video_predictions) * 100

                format_str = ('%s: step %s, accuracy = %.2f%%\n')
                print (format_str % (datetime.now(), global_step, test_accuracy))
                
                summary = sess.run(summary_op)
                test_summary = tf.Summary()
                test_summary.ParseFromString(summary)
                test_summary.value.add(tag='accuracy', simple_value=test_accuracy)
                test_summary_writer.add_summary(test_summary, global_step)

                # When done, ask the threads to stop.
                coord.request_stop()

        except Exception as e:
            coord.request_stop(e)


        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    run()     
