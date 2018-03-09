import tftables
import tensorflow as tf

with tf.device('/cpu:0'):
    # This function preprocesses the batches before they
    # are loaded into the internal queue.
    # You can cast data, or do one-hot transforms.
    # If the dataset is a table, this function is required.
    def input_transform(tbl_batch):
        labels = tbl_batch['label']
        data = tbl_batch['data']

        truth = tf.to_float(tf.one_hot(labels, num_labels, 1, 0))
        data_float = tf.to_float(data)

        return truth, data_float

    # Open the HDF5 file and create a loader for a dataset.
    # The batch_size defines the length (in the outer dimension)
    # of the elements (batches) returned by the reader.
    # Takes a function as input that pre-processes the data.
    loader = tftables.load_dataset(filename='bottleneck_fc_model.h5',
                                   dataset_path='/data',
                                   input_transform=input_transform,
                                   batch_size=16)

# To get the data, we dequeue it from the loader.
# Tensorflow tensors are returned in the same order as input_transformation
truth_batch, data_batch = loader.dequeue()

# The placeholder can then be used in your network
result = my_network(truth_batch, data_batch)

with tf.Session() as sess:

    # This context manager starts and stops the internal threads and
    # processes used to read the data from disk and store it in the queue.
    with loader.begin(sess):
        for _ in range(num_iterations):
            sess.run(result)