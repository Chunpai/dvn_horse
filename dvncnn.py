#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
from PIL import Image
sys.path.append(os.path.abspath('/network/rit/lab/ceashpc/chunpai/PycharmProjects/cnn_dvn'))
import tensorflow as tf
import time
from data_helper import *
import Queue
import threading
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class DVCNN(object):
    def __init__(self, data_dir, height, width, channel, label_dim, one_hot_label_dim, weight_decay, learning_rate,
                 inf_lr, inf_iter):

        self.sentinel = object()
        self.height = height
        self.width = width
        self.channel = channel
        self.label_dim = label_dim
        self.one_hot_label_dim = one_hot_label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.inf_lr = inf_lr
        self.inf_iter = inf_iter
        self.weights = []
        self.loss = 0.  # loss
        self.current_step = 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build() # construct computation graph
        # self.saver = tf.train.Saver()

        config = tf.ConfigProto(device_count = {"GPU":1})
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()

        print("initial value of W1 {}".format(self.sess.run(self.W1)))


    def load(self):
        self.saver = tf.train.import_meta_graph("../tmp/model.ckpt.meta")
        self.saver.restore(self.sess, "../tmp/model.ckpt")

        W1 = self.sess.run('fcn/W1:0')
        b1 = self.sess.run('fcn/b1:0')
        self.W1 = self.W1[:,:,:3,:].assign(W1)
        self.b1 = self.b1.assign(b1)
        W2 = self.sess.run('fcn/W2:0')
        b2 = self.sess.run('fcn/b2:0')
        self.W2 = self.W2[:,:,:3,:].assign(W2)
        self.b2 = self.b2.assign(b2)
        W3 = self.sess.run('fcn/W3:0')
        b3 = self.sess.run('fcn/b3:0')
        self.W3 = self.W3[:,:,:3,:].assign(W3)
        self.b3 = self.b3.assign(b3)
        print("loaded value of W1 {}".format(self.sess.run(self.W1)))

    def build(self):
        with tf.name_scope('input'):
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.height, self.width, self.channel],
                                           name='features')
            # self.features = tf.layers.batch_normalization(self.features)
            # we will feed in the adversary labels and ground truth value to train the model
            self.labels = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.height, self.width, self.label_dim],
                                         name='labels')
            self.one_hot_labels = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self.height, self.width, self.one_hot_label_dim],
                                                 name='one_hot_labels')
            # self.gt_values = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='gt_values')
            self.gt_values = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='gt_values')
            self.dropout = tf.placeholder_with_default(0., shape=(), name='dropout')
            self.inputs = tf.concat([self.features, self.labels], axis=3, name='input_concat')

        # Question:
        # Is it necessary to add activation function for CNN layers? Try
        with tf.variable_scope('cnn'):
            # input dim: m, height, width, channel + one_hot_label_dim = m, 12, 12, 3+2
            # output dim: m, 24, 24, 64
            # TODO do we need bias in the convolution ?
            filter_shape = [5, 5, 4, 64]
            self.W1 = tf.get_variable(name="W1", shape=filter_shape, initializer= tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.constant(0, shape=[filter_shape[3]], dtype=tf.float32), name="b1")
            conv1 = tf.nn.conv2d(self.inputs, self.W1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
            self.conv_1 = tf.nn.relu(tf.nn.bias_add(conv1, self.b1), name="relu1")
            self.weights.append(self.W1)
            self.weights.append(self.b1)

            filter_shape = [5, 5, 64, 128]
            # self.W2 = tf.get_variable(tf.truncated_normal(shape=filter_shape, stddev=0.01), name="W2")
            self.W2 = tf.get_variable(name="W2", shape=filter_shape, initializer= tf.contrib.layers.xavier_initializer())
            self.b2 = tf.Variable(tf.constant(0, shape=[filter_shape[3]], dtype=tf.float32), name="b2")
            conv2 = tf.nn.conv2d(self.conv_1, self.W2, strides=[1, 2, 2, 1], padding="SAME", name="conv2")
            self.conv_2 = tf.nn.relu(tf.nn.bias_add(conv2, self.b2), name="relu2")
            self.weights.append(self.W2)
            self.weights.append(self.b2)

            filter_shape = [5, 5, 128, 128]
            # self.W3 = tf.get_variable(tf.truncated_normal(shape=filter_shape, stddev=0.01), name="W3")
            self.W3 = tf.get_variable(name="W3", shape=filter_shape, initializer= tf.contrib.layers.xavier_initializer())
            self.b3 = tf.Variable(tf.constant(0, shape=[filter_shape[3]], dtype=tf.float32), name="b3")
            conv3 = tf.nn.conv2d(self.conv_2, self.W3, strides=[1, 2, 2, 1], padding="SAME", name="conv3")
            self.conv_3 = tf.nn.relu(tf.nn.bias_add(conv3, self.b3), name="relu3")
            self.weights.append(self.W3)
            self.weights.append(self.b3)



        with tf.variable_scope('full'):
            # output dim: m, 6x6x128
            conv3_flat = tf.reshape(self.conv_3, [-1, 6 * 6 * 128])
            dropout = tf.nn.dropout(conv3_flat, keep_prob=1 - self.dropout, name='dropout')
            # output dim: m, 384
            dense_1 = tf.layers.dense(inputs=dropout,
                                      units=384,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
                                      use_bias=True,
                                      name='dense_1')
            dense_2 = tf.layers.dense(inputs=dense_1,
                                      units=192,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
                                      use_bias=True,
                                      name='dense_2')
            self.raw_prediction = tf.layers.dense(inputs=dense_2, units=1, use_bias=True, name='output')
            self.pred_val = tf.nn.sigmoid(self.raw_prediction, name='predicted_value')

        with tf.name_scope('loss'):
            # TODO add regularization
            self.regularization_losses = tf.losses.get_regularization_loss()
            self.loss += self.regularization_losses

            self.loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt_values,
                                                                 logits=self.raw_prediction,
                                                                 name='cross_entropy')
            self.loss += self.loss2
            self.mean_loss = tf.reduce_mean(self.loss)

        with tf.name_scope('gradients'):
            # used to maximize the predicted value
            self.value_gradient = tf.gradients(self.pred_val, self.labels)[0]
            self.raw_pred_gradient = tf.gradients(self.raw_prediction, self.labels)[0]
            # used to minimize the loss or maximize the loss
            self.loss_gradient = tf.gradients(self.mean_loss, self.labels)[0]

        with tf.name_scope('train'):
            # TODO we need to try different optimizer
            # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.mean_loss,
            #                                                                                  global_step=self.global_step,
            #                                                                                  name='train_op')
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mean_loss,
                                                                                  global_step=self.global_step,
                                                                                  name='train_op')

    def train(self, train_features, train_labels, train_labels_full, test_features, test_labels, test_labels_full,
              epochs, batch_size, dropout, proportion, output_file):

        f = open("{}/{}.txt".format(output_file, output_file), "a")
        for epoch in range(epochs):
            epoch_generated_values = []
            start_time = time.time()
            batches = batch_iter(train_features, train_labels, batch_size=batch_size, num_epochs=1, shuffle=True)
            for i, batch in enumerate(batches):
                features = batch[0]
                gt_labels = batch[1]
                generated_labels, generated_values = self.generate_examples(features, gt_labels, proportion, train=True)
                # print("epoch:{}, batch: {}, generated mean values: {:.6f}, std: {:.6f}".format(epoch, i, np.mean(generated_values), np.std(generated_values)))
                for value in generated_values:
                    if value != 1.0:
                        epoch_generated_values.append(value)
                self.current_step, _, output_values, loss, train_mean_loss, reg_loss = self.sess.run(
                    [self.global_step,
                     self.train_step,
                     self.pred_val,
                     self.loss,
                     self.mean_loss,
                     self.regularization_losses],
                    feed_dict={
                        self.features: features,
                        self.labels: generated_labels,
                        self.gt_values: generated_values,
                        self.dropout: dropout}
                )

            f.write("Epoch: {}, Train Mean Loss: {}\n".format(epoch, train_mean_loss))
            print("Epoch: {}, Train Mean Loss: {}, Regularization Loss {}".format(epoch, train_mean_loss, reg_loss))
            print("Generated Min Value: {}, Mean Value: {}, Max Value: {}".format(np.min(epoch_generated_values), np.mean(epoch_generated_values), np.max(epoch_generated_values) ))
            if epoch % 1000 == 0:
                self.saver.save(self.sess, "../tmp/model_{}.ckpt".format(epoch))
                """
                train_iou_scores = []
                intersect_list = []
                union_list = []
                for idx in range(0, len(train_features), 36):
                    features = train_features[idx:min(len(train_features), idx + 36)]
                    gt_labels = train_labels[idx:min(len(train_labels), idx + 36)]
                    gt_labels_full = train_labels_full[idx / 36]
                    pred_labels, train_values = self.generate_examples(features, gt_labels, proportion, train=False)
                    print("image {}, generated mean values: {:.6f}, std: {:.6f}".format(idx/36, np.mean(train_values), np.std(train_values)))

                    pred_labels_full = recover_full_resolution(pred_labels, gt_labels_full)
                    combo = np.array([pred_labels_full, gt_labels_full])
                    intersect = np.sum(np.min(combo, axis=0))
                    union = np.sum(np.max(combo, axis=0))
                    intersect_list.append(intersect)
                    union_list.append(union)
                    iou_score = intersect * 1.0 / union
                    train_iou_scores.append(iou_score)
                train_iou_scores = np.array(train_iou_scores)
                print("Train Mean IOU: {:.4f}, Train Global IOU: {:.4f}".format(np.mean(train_iou_scores), np.sum(intersect_list) / float(np.sum(union_list))))
                f.write("Train Mean IOU: {:.4f}, Train Global IOU: {:.4f}\n".format(np.mean(train_iou_scores), np.sum(intersect_list) / float(np.sum(union_list))))
                """

            if epoch % 10 == 0:
                test_iou_scores = []
                intersect_list = []
                union_list = []
                for idx in range(0, len(test_features), 36):
                    features = test_features[idx:min(len(test_features), idx + 36)]
                    gt_labels = test_labels[idx:min(len(test_labels), idx + 36)]
                    gt_labels_full = test_labels_full[idx / 36]
                    pred_labels, test_values = self.generate_examples(features, gt_labels, proportion, train=False)

                    pred_labels_full = recover_full_resolution(pred_labels, gt_labels_full)
                    combo = np.array([pred_labels_full, gt_labels_full])
                    intersect = np.sum(np.min(combo, axis=0))
                    union = np.sum(np.max(combo, axis=0))
                    intersect_list.append(intersect)
                    union_list.append(union)
                    iou_score = intersect * 1.0 / union
                    test_iou_scores.append(iou_score)
                    if idx == 0:
                        print("true mask:")
                        print_mask(gt_labels_full)
                        print("pred mask:")
                        print_mask(pred_labels_full)

                        img = np.reshape(pred_labels_full, (32, 32))
                        img[img > 0] = 255
                        im = Image.fromarray(np.uint8(img))
                        im.save("{}/pred_mask_{}.png".format(output_file, epoch), format("PNG"))
                        # im.close()
                test_iou_scores = np.array(test_iou_scores)
                f.write("Test  Mean IOU: {:.4f}, Test  Global IOU: {:.4f}\n".format(np.mean(test_iou_scores), np.sum(intersect_list) / float(np.sum(union_list))))
                print("Test  Mean IOU: {:.4f}, Test Global IOU: {:.4f}".format(np.mean(test_iou_scores), np.sum(intersect_list) / float(np.sum(union_list))))
                print("Test Min IOU: {:.4f}, Index: {}".format(np.min(test_iou_scores), np.argmin(test_iou_scores)))
                print("Test Max IOU: {:.4f}, Index: {}".format(np.max(test_iou_scores), np.argmax(test_iou_scores)))
                f.write("Time Cost: {}\n\n".format(time.time() - start_time))
                print("Time Cost: {}\n".format(time.time() - start_time))
        f.close()


    def generate_examples(self, features, gt_labels, proportion=0.5, train=False):
        """generate adversary samples based on features and ground truth label"""
        init_labels = self.get_initialization(features)
        # In training, generate adversarial examples 50 % of the time
        rand_num = np.random.rand()
        # print("gt_labels shape", gt_labels.shape)
        if train and rand_num > proportion:
            # print("Ground Truth Samples")
            # generated_labels = gt_labels

            print("Adversary Samples")
            # maximize loss, same as generate the adversarial examples
            gt_indices = np.random.rand(gt_labels.shape[0]) > 0.5
            init_labels[gt_indices] = gt_labels[gt_indices]
            generated_labels = self.inference(features,
                                              init_labels,
                                              learning_rate=self.inf_lr,
                                              gt_labels=gt_labels,
                                              num_iterations=3)
        else:
            # print("Inference Samples")
            # print("generating standard examples...")
            # maximize predicted values, which is standard inference to generate examples
            generated_labels = self.inference(features,
                                              init_labels,
                                              learning_rate=self.inf_lr,
                                              num_iterations=self.inf_iter,
                                              train=train)

            # TODO get correct gt_values
        generated_values = np.zeros(([gt_labels.shape[0], 1]))
        generated_values = [iou(np.expand_dims(generated_labels[idx], axis=0), np.expand_dims(gt_labels[idx], axis=0))
                            for idx in np.arange(0, gt_labels.shape[0])]
        generated_values = np.array(generated_values).reshape([gt_labels.shape[0], 1])
            # return the generated labeled, and corresponding ground-truth value for this label
        return generated_labels, generated_values


    def inference(self, features, init_labels, learning_rate, gt_labels=None, num_iterations=100, train=False):
        # for adversary inference, the init_labels is not all zeros, some are gt_labels
        pred_labels = init_labels
        binary_pred_labels = init_labels
        if gt_labels is not None:
            gt_scores = np.zeros([gt_labels.shape[0], 1])

        avg_pred_values = []
        for idx in range(num_iterations):
            if gt_labels is not None:  # maximize loss to generate the adversarial examples
                gt_scores = [iou(np.expand_dims(pred_labels[i], axis=0), np.expand_dims(gt_labels[i], axis=0))
                             for i in np.arange(gt_labels.shape[0])]
                gt_scores = np.array(gt_scores).reshape([gt_labels.shape[0], 1])

                # here the grad is loss_gradient, we going to maximize it using gradient ascent
                loss, pred_val, grad = self.sess.run([self.mean_loss, self.pred_val, self.loss_gradient],
                                                     feed_dict={self.features: features,
                                                                self.labels: pred_labels,
                                                                self.gt_values: gt_scores})
                # print("iteration:{}, predicted value: {}, loss:{}".format(idx, pred_val, loss))

            else:  # maximize predicted values to get the best predicted labels
                pred_val, grad = self.sess.run([self.pred_val, self.value_gradient],
                                               feed_dict={self.features: features,
                                                          self.labels: pred_labels})

                # np.set_printoptions(precision=20)

                # if train == False:
                #     print("iteration:{}, predicted value:{}".format(idx, pred_val))
                #     print("gradient:\n{}".format(grad))
                # print("iteration:{}, predicted value:{}".format(idx, pred_val))

            if idx > 30 and np.mean(pred_val) <= np.mean(avg_pred_values[-10:]):
                learning_rate *= 0.5
            avg_pred_values.append(np.mean(pred_val))
            # pred_labels = softmax(np.log(pred_labels) + learning_rate * grad)
            # print("pred labels {}".format(pred_labels))
            pred_labels = pred_labels + learning_rate * grad
            pred_labels[pred_labels < 0] = 0.
            pred_labels[pred_labels > 1] = 1.
            # binary_pred_labels[pred_labels >= 0.5] = 1
            # binary_pred_labels[pred_labels < 0.5] = 0
            # pred_labels = softmax(pred_labels)
            # print("binary pred labels", binary_pred_labels[0, :, :, 0])
        return pred_labels

    def get_initialization(self, features):
        # return np.zeros([features.shape[0], self.height, self.width, self.label_dim]) + 0.5
        return np.zeros([features.shape[0], self.height, self.width, self.label_dim])

    def _generator_queue(self, train_features, train_labels, batch_size, num_threads=1):
        queue = Queue.Queue(maxsize=100)

        # Build indices queue to ensure unique use of each batch
        indices_queue = Queue.Queue()
        for idx in np.arange(0, len(train_features), batch_size):
            indices_queue.put(idx)

        def generate():
            try:
                while True:
                    # Get a batch
                    idx = indices_queue.get_nowait()
                    features = train_features[idx: min(len(train_features), idx + batch_size)]
                    gt_labels = train_labels[idx:min(len(train_labels), idx + batch_size)]

                    # Generate data (predicted labels and their true performance)
                    pred_labels, iou_scores = self.generate_examples(features, gt_labels=gt_labels, train=True)
                    queue.put((features, pred_labels, iou_scores))
            except Queue.Empty:
                queue.put(self.sentinel)

        for _ in range(num_threads):
            thread = threading.Thread(target=generate)
            thread.start()

        return queue

    def predict(self, features, binarize=True, num_iterations=20):
        init_labels = self.get_initialization(features)
        features = np.array(features, np.float64)

        features -= self.mean[0]
        features /= self.std[0]

        labels = self.inference(features.reshape([1, self.height, self.width, 3]), init_labels,
                                learning_rate=self.inf_lr, num_iterations=num_iterations)

        if binarize:
            return labels >= 0.5
        else:
            return labels
