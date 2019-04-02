from tensorflow.python.client import device_lib
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print(device_lib.list_local_devices())
import tensorflow as tf



def check_variables(file_path):
    print_tensors_in_checkpoint_file(file_name=file_path, all_tensors=True, tensor_name='')


def restore_variables():
    saver = tf.train.import_meta_graph("../tmp/model.ckpt.meta")
    # W2 = tf.Variable(tf.ones(shape=[5,5,4,64]))
    W2 = tf.get_variable("W2", shape=[5,5,4,64], initializer=tf.contrib.layers.xavier_initializer())
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        # saver.restore(sess, tf.train.latest_checkpoint("../tmp/"))
        saver.restore(sess, "../tmp/model.ckpt")
        W1 = sess.run('fcn/W1:0')
        print sess.run(init)
        print W1.shape
        print sess.run(W2)
        W2 = W2[:,:,:3,:].assign(W1)
        print sess.run(W2)



if __name__ == '__main__':
    restore_variables()


