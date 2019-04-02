from dvncnn import DVCNN
import tensorflow as tf
from data_helper import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.0001, "initial learning rate")
flags.DEFINE_integer("epochs", 10001, "num. of epoches to train")
flags.DEFINE_float("dropout", 0.25, "drop out rate (1 - keep probability)")
flags.DEFINE_float("weight_decay", 0.001, "weight for l2 loss on embedding matrix")
flags.DEFINE_integer("early_stopping", 8, "tolerance for early stopping (# of epoches).")
flags.DEFINE_integer("label_dim", 1, "dimension of label")
flags.DEFINE_integer("one_hot_label_dim", 2, "dimension of one hot label")
flags.DEFINE_float("inf_lr", 5e5, "learning rate for inference")
flags.DEFINE_integer("inf_iter", 30, "iterations for inference")
flags.DEFINE_integer("batch_size", 50, "batch size for training")
flags.DEFINE_float("proportion", 0.5, "proportion of inference and adversarial")
flags.DEFINE_string("output_file", "output2", "output file name")

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
print("Learning rate: {}\nWeight Decay: {}\nInference Learning rate: {}"
        "\nOutput File:{}\nSeed:{}\n".format(FLAGS.learning_rate, FLAGS.weight_decay,
                                             FLAGS.inf_lr, FLAGS.output_file, seed))

f = open("{}/{}.txt".format(FLAGS.output_file, FLAGS.output_file), "a")
f.write("Learning rate: {}\nWeight Decay: {}\nInference Learning rate: {}"
      "\nOutput File:{}\nSeed:{}\n".format(FLAGS.learning_rate, FLAGS.weight_decay,
                                           FLAGS.inf_lr, FLAGS.output_file, seed))

train_imgs, test_imgs, train_masks, test_masks = load_pure_dataset()
print "all data splits", train_imgs.shape, train_masks.shape, test_imgs.shape, test_masks.shape
train_masks[train_masks>0] = 1
test_masks[test_masks>0] = 1
train_imgs_cropped, train_masks_cropped = data_augmentation_test(train_imgs, train_masks)
test_imgs_cropped, test_masks_cropped = data_augmentation_test(test_imgs, test_masks)
print "train data", train_imgs_cropped.shape, train_masks_cropped.shape
print "test data", test_imgs_cropped.shape, test_masks_cropped.shape

# normalization
train_imgs_cropped = np.array(train_imgs_cropped, dtype=np.float32)
mean = np.mean(train_imgs_cropped, axis=0)
std = np.std(train_imgs_cropped, axis=0)
train_imgs_cropped -= mean
train_imgs_cropped /= std
test_imgs_cropped = np.array(test_imgs_cropped, dtype=np.float32)
test_imgs_cropped -= mean
test_imgs_cropped /= std


train_size, height, width, channel = train_imgs_cropped.shape
_, _, _, label_dim = train_masks_cropped.shape


data_dir = '/network/rit/lab/ceashpc/chunpai/PycharmProjects/cnn_dvn/tmp/'
model = DVCNN(data_dir, height, width, channel,
              label_dim=FLAGS.label_dim,
              one_hot_label_dim=FLAGS.one_hot_label_dim,
              weight_decay=FLAGS.weight_decay,
              learning_rate=FLAGS.learning_rate,
              inf_lr=FLAGS.inf_lr,
              inf_iter=FLAGS.inf_iter)

model.load()  # load the pretrain parameters


f.write("adversarial + inference (0.0 init) + adam + seed=0")
model.train(train_imgs_cropped,
            train_masks_cropped,
            train_masks,
            test_imgs_cropped,
            test_masks_cropped,
            test_masks,
            epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            dropout=FLAGS.dropout,
            proportion=FLAGS.proportion,
            output_file=FLAGS.output_file)
