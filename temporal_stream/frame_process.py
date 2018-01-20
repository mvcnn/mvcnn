import tensorflow as tf
import os
import numpy as np  # linear algebra
import math
import pickle

from sklearn.preprocessing import normalize


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "labels": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "rgb": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "audio": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    return image


def get_all_records(frame_lvl_record):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([frame_lvl_record], num_epochs=1)
        image = read_and_decode(filename_queue)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while True:
                example = sess.run([image])
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


# extract frames from tfrecords file
def extract_frames(frame_lvl_record):
    feat_rgb = []
    feat_audio = []
    labels = []
    num_frames = 0
    for idx, example in enumerate(tf.python_io.tf_record_iterator(frame_lvl_record)):
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        labels.append(tf_seq_example.context.feature['labels'].int64_list.value)
        sess = tf.InteractiveSession()
        rgb_frame = []
        audio_frame = []
        # iterate through frames
        for i in range(n_frames):
            rgb_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
            audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())

        sess.close()
        feat_rgb.append(rgb_frame)
        feat_audio.append(audio_frame)

        print('The %ith video has %d frames and labels %s' % (idx, len(rgb_frame), str(labels[idx])))
        num_frames += len(rgb_frame)

    return feat_rgb, feat_audio, num_frames, labels


def fisher_vector(gmm, data):
    # compute posterior
    Q = gmm.predict_proba(data)

    # compute mean and covariance derivatives
    N = np.size(data, 0)

    # sum the rows of posterior matrix
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, data) / N
    Q_xx_2 = np.dot(Q.T, data ** 2) / N
    d_mu = Q_xx - Q_sum * gmm.means_
    d_cov = (- Q_xx_2 - Q_sum * gmm.means_ ** 2 + Q_sum * gmm.covariances_ + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector and normalize.
    return normalize(np.hstack((d_mu.flatten(), d_cov.flatten())))


def pca_proj(mat, epsilon):
    m = np.size(mat, axis=0)
    n = np.size(mat, axis=1)

    # normalize X by subtracting the mean
    mu = np.mean(mat, axis=1)
    mat = mat - np.expand_dims(mu, axis=1)
    print(mat.shape)

    if m < n:
        A = np.matmul(mat, mat.T)
        U, S, _ = np.linalg.svd(A)
    else:
        A = np.matmul(mat.T, mat)
        print(A.shape)
        V, S, _ = np.linalg.svd(A)
        U = np.matmul(np.matmul(mat, V), np.diag(1. / (np.sqrt(np.add(S, epsilon)))))
    return U, S, n


def imPCAwhiten(mat, U, S, epsilon, dim, n):
    mu = np.mean(mat, axis=1)
    mat = mat - np.expand_dims(mu, axis=1)

    matRot = np.matmul(U.T, mat)
    matWh = math.sqrt(n) * np.matmul(np.diag(1. / np.sqrt(np.add(S, epsilon))), matRot)
    matWh = matWh[:dim, :]

    # L2 normalize
    normalize(matWh, axis=0)
    return matWh


def write_to_tfrecord(mat, labels):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
