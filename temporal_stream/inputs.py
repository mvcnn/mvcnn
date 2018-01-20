import cPickle
import random
import tensorflow as tf

# CPU queue parameters
BATCH_PER_FILE = 1
NUM_THREADS = 8


class Temporal_Input(object):
    def __init__(self, data_dir, batch_size, image_size, temporal_depth, 
                 num_test_crops_per_sample = 6, num_test_samples_per_file = 2):
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_dir = data_dir
        self.temporal_depth = temporal_depth
        self.num_test_crops_per_sample = num_test_crops_per_sample
        self.num_test_samples_per_file = num_test_samples_per_file

    def unpickle(self, file):
        '''
        Unpickle file
        :param file: file (string)
        :return: file contents (dict)
        '''
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def _generate_image_and_label_batch(self, example_list, min_queue_examples, shuffle):
        '''
        Generate a batch of input and label pairs
        :param example_list: list of (input tensor, label tensor) tuples
        :param min_queue_examples: integer for queue capacity
        :param shuffle: flag to shuffle inputs (bool)
        :return: data object containing inputs, labels and input summaries
        '''
        if shuffle:
            # min_after_dequeue defines how big a buffer we will randomly sample
            #   from -- bigger means better shuffling but slower start up and more
            #   memory used.
            images, label_batch = tf.train.shuffle_batch_join(
                example_list,
                batch_size=self.batch_size,
                capacity=min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=2*self.batch_size)
        else:
            images, label_batch = tf.train.batch_join(
                example_list,
                batch_size=self.batch_size,
                enqueue_many = True,
                capacity=min_queue_examples) #+ 3 * self.batch_size)

        # Display the training images in the visualizer.
        image_summary = tf.summary.image('images', images[:,:,:,:3])

        class Data(object):
            pass

        data = Data()
        data.output = images
        data.labels = tf.reshape(label_batch, [self.batch_size])
        data.shape = images.get_shape().as_list()
        data.image_summary = image_summary

        return data   


    def read(self, filename):
        '''
        Read bin file containing motion vector frames
        :param filename: filename of dequeued file
        :return: mv_record tensor (num_frames, height*width), height of frames, width of frames, video label tensor
        '''
        #for P-frames read entire file and take random slice of 10 frames from matrix         
        # Read a record, getting filenames from the filename_queue. 
        value = tf.read_file(filename)
        
        height_bytes = 4
        width_bytes = 4
        label_bytes = 4
        header_bytes = width_bytes + height_bytes + label_bytes
        
        # Convert from a string to a vector of int8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.int8)

 
        # The first bytes represent xdim, ydim, label, which we convert from int8->int32.
        width = tf.bitcast(tf.slice(record_bytes, [0], [width_bytes]), tf.int32)
        height = tf.bitcast(tf.slice(record_bytes, [width_bytes], [height_bytes]), tf.int32)
        label = tf.bitcast(tf.slice(record_bytes, [width_bytes+height_bytes], [label_bytes]), tf.int32)

        mv_bytes = tf.multiply(height, width)
        record = record_bytes[header_bytes:]
        mv_record = tf.reshape(record, [-1, mv_bytes])

        
        return mv_record, height, width, label
        
          
    def preprocess(self, mv_record, height, width, label, mode='train', iteration=None):
        '''
        Preprocess mv_record from dequeued file and generate input
        :param mv_record: mv_record tensor (num_frames, height*width)
        :param height: height of frames
        :param width: width of frames
        :param label: video label tensor
        :param mode: process mode; 'train', 'test' or 'val' (string)
        :param iteration: preprocess iteration for current file (int) - controls starting frame index for input generation
        :return: (image tensor, label tensor) tuple for input. image contains the generated input (height/2 x width x 2*temporal_depth)
        '''
        mv_record_shape = tf.shape(mv_record)
        mv_bytes = tf.multiply(height, width)

        if mode == 'train':
            # get a random starting index between 0 and record_shape[0]
            rand_idx = tf.random_uniform([], minval=0, maxval=mv_record_shape[0], dtype=tf.int32)
        elif mode == 'test' or mode == 'val':
            # select the starting index based on current iteration
            rand_idx = mv_record_shape[0] - mv_record_shape[0]/(iteration+1) 
            #tf.minimum(config.TEMPORAL_CHUNK_SIZE*iteration, mv_record_shape[0])
        mv_slice = tf.slice(mv_record, [rand_idx, 0], [-1, mv_bytes])
        num_tiles = tf.cast(tf.ceil(tf.truediv(self.temporal_depth, mv_record_shape[0])), tf.int32)
        mv_tiled = tf.tile(tf.cast(mv_record, tf.int32), [num_tiles, 1])
        mv_input = tf.concat(axis=0, values=[tf.cast(mv_slice, tf.int32), mv_tiled])
        mv_input = mv_input[:self.temporal_depth,:]

        mv_input = tf.reshape(mv_input, [-1])

        
        #split each list element into x and y components and stack depth wise
        image_shape = tf.stack([2*self.temporal_depth, tf.div(height,2), width])
        image_chunk = tf.reshape(mv_input, image_shape)
        image_chunk = tf.transpose(image_chunk, perm=[1, 2, 0])
        
        #cast image block as float
        image_chunk = tf.cast(image_chunk, tf.float32)


        def normalize_images(image_chunk):
            #mean subtraction and normalization
            mean, variance = tf.nn.moments(image_chunk, axes=[0,1])
            float_image = tf.subtract(image_chunk,tf.expand_dims(tf.expand_dims(mean,0),0))
            #float_image = tf.image.per_image_standardization(image_chunk)
            
            return float_image
       

        if mode == 'train': 
            #random horizontal flip
            resized_image_chunk = tf.image.random_flip_left_right(image_chunk)

            #jitter input scale
            ratios = [0.5, 0.667, 0.833, 1]
            opts = [int(self.image_size * r) for r in ratios]
            target_width = target_height = random.choice(opts)

            # Random crop portion with dims [config.IMAGE_SIZE x config.IMAGE_SIZE x 2*config.TEMPORAL_CHUNK_SIZE]            
            resized_image_chunk = tf.random_crop(resized_image_chunk, [target_height, target_width, 2*self.temporal_depth])
            
            #Resize to config.IMAGE_SIZE
            resized_image_chunk = tf.image.resize_images(resized_image_chunk, [self.image_size, self.image_size])

            float_image = normalize_images(resized_image_chunk)
            return float_image, label
            
        else:
            #Crop central portion of image and four corners
            center_crop = tf.image.resize_image_with_crop_or_pad(image_chunk, self.image_size, self.image_size)
            label = tf.expand_dims(label, -1)

            if mode == 'test':
                top_left_crop = tf.image.crop_to_bounding_box(image_chunk, 0, 0, self.image_size, self.image_size)
                top_right_crop = tf.image.crop_to_bounding_box(image_chunk, 0, tf.subtract(width, self.image_size), 
                                                               self.image_size, self.image_size)
                bottom_right_crop = tf.image.crop_to_bounding_box(image_chunk, tf.subtract(tf.div(height,2), self.image_size), 
                                                                  tf.subtract(width,self.image_size), self.image_size, 
                                                                  self.image_size)
                bottom_left_crop = tf.image.crop_to_bounding_box(image_chunk, tf.subtract(tf.div(height,2), self.image_size), 
                                                                 0, self.image_size, self.image_size)
                
                center_crop_f = tf.image.flip_left_right(center_crop)

                all_crops = [center_crop, center_crop_f, top_left_crop, top_right_crop, bottom_right_crop, bottom_left_crop]
            
                normalized_crops = []
                for crop in all_crops[:self.num_test_crops_per_sample]:
                    normalized_crops.append(normalize_images(crop))
                
                float_image = tf.stack(normalized_crops)               
                label = tf.concat(axis=0, values=[label]*self.num_test_crops_per_sample)
            elif mode == 'val':
                float_image = tf.stack([normalize_images(center_crop)])
            
            return float_image, label

        
    
    def generate_batches_to_train(self, filenames):
        '''
        Generate batches to train
        :param filenames: list of files names to train
        :return: function to generate batches from list of (image, label) tuples
        '''
         # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_threads = NUM_THREADS
        
        example_list = []
        for _ in range(read_threads):
            filename = filename_queue.dequeue()
            r, h, w, l = self.read(filename)
            for _ in range(BATCH_PER_FILE):
                example_list.append(self.preprocess(r, h, w, l))

        # Ensure that the random shuffling has good mixing properties.
        # min_fraction_of_examples_in_queue = 0.01
        min_queue_examples = 5000
        print ('Filling queue with %d inputs before starting to train. '
             'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(example_list,
                                             min_queue_examples, shuffle=True)
        
        
    
    def generate_batches_to_eval(self, filenames, test = True):
        '''
        Generate batches for validation/test
        :param filenames: list of files names for validation/test
        :return: function to generate batches from list of (image, label) tuples
        '''
        #condition on whether to shuffle files in queue
        shuffle = False if test else True
        mode = 'val' if shuffle else 'test'

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

        #number of iterations to run single video over
        num_iter = self.num_test_samples_per_file if test else 1
        filename = filename_queue.dequeue()
        
        # Read examples from files in the filename queue.
        float_image, label = [], []
        for n in range(num_iter):
            r, h, w, l = self.read(filename)
            x, y = self.preprocess(r, h, w, l, mode=mode, iteration=n)
            label.append(y)
            float_image.append(x)
        
        label = tf.concat(axis=0, values=label)
        float_image = tf.concat(axis=0, values=float_image)
        example_list = [(float_image, label)]

        # Ensure that the random shuffling has good mixing properties.
        #min_fraction_of_examples_in_queue = 0.01
        min_queue_examples = 5000
        

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(example_list,
                                             min_queue_examples, shuffle=False)
