import os
import glob
import struct
import sys
import numpy as np
import tensorflow as tf

# Command line options
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', os.path.abspath('data'),
                           """Data directory""")
tf.app.flags.DEFINE_string('out_dir', os.path.abspath('split'),
                           """Directory to write new bins""")
tf.app.flags.DEFINE_string('path_to_class_list', os.path.join(os.path.curdir, 'ucfTrainTestlist', 'classInd.txt'),
                           """Path to class index list""")
tf.app.flags.DEFINE_string('path_to_file_list', os.path.join(os.path.curdir, 'ucfTrainTestlist', 'trainlist01.txt'),
                           """Path to list of filenames to parse (set to 'None' to train on everything in data_dir)""")


# get all filenames in directory
data_files = glob.glob(os.path.join(FLAGS.data_dir, '*.bin'))

# make out_dir if it doesn't exist
if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

# Get videos to write from txt file (UCF-101)
path = FLAGS.path_to_file_list
if path != None:
    print('Parsing inputs: ')
    with open(path) as f:
        files_to_parse = [os.path.splitext(line.split()[0].split('/')[1])[0] for line in f]

    finput = [d for d in data_files for item in files_to_parse if item in d]
else:
    finput = data_files

# read labels into dictionary
label_dict = {}
with open(FLAGS.path_to_class_list) as f:
    for line in f:
        (val, key) = line.split()
        label_dict[key.lower()] = int(val) - 1

frame_count = {}
for file in finput:
    file_header = os.path.basename(file).split('.')[0]

    # get label index
    fileparts = file_header.split('_')
    label = fileparts[-3]
    label_idx = int(label_dict[label.lower()])

    with open(file, 'rb') as f:
        # read rest of file
        data = f.read()

    if len(data) == 0:
        continue

    size = struct.unpack('<IIII', data[0:16])

    # width and height of the image
    pts = int(size[0])
    frameIndex = int(size[1])
    width = int(size[2])
    height = int(size[3])

    frame_bytes = 16 + 1 + (height * width)
    bin_file = os.path.join(FLAGS.out_dir, file_header + '.bin')

    print(file_header, height, width, len(data))

    # label to bytes
    label = struct.pack('i', label_idx)

    with open(bin_file, 'ab') as b:
        bytes = data[8:16] + label
        b.write(bytes)

    count = 0
    for i in range(0, len(data), frame_bytes):
        # <int PTS> <int frameIndex> <int xdim> <int ydim>  <char frameType> <int DATA>
        size = struct.unpack('<IIII', data[i:i + 16])

        # width and height of the image
        width = int(size[2])
        height = int(size[3])

        # frame_type
        frame_type = struct.unpack("c", data[i + 16:i + 17])
        frame_type = frame_type[0]

        # motion vector
        format = ('%ib' % height * width)
        # mv_raw = struct.unpack(format, data[i+17:i+frame_bytes])
        mv_raw = np.asarray(struct.unpack(format, data[i + 17:i + frame_bytes]))

        mv_bytes = data[i + 17:i + frame_bytes]

        if frame_type == 'P':
            with open(bin_file, 'ab') as b:
                bytes = mv_bytes
                b.write(bytes)
            count += 1

        frame_count[os.path.basename(bin_file.strip('\n'))] = count
