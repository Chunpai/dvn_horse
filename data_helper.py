import numpy as np
import pickle
from PIL import Image


def load_pure_dataset():
    # path = '/network/rit/lab/ceashpc/chunpai/PycharmProjects/cnn_dvn/horse/'
    train_fn = 'train/train_imgs.pkl'
    train_masks = 'train/train_masks.pkl'
    test_fn = 'test/test_imgs.pkl'
    test_masks = 'test/test_masks.pkl'

    train_imgs = pickle.load(open(train_fn, "rb"))
    test_imgs = pickle.load(open(test_fn, "rb"))
    train_masks = pickle.load(open(train_masks, "rb"))
    test_masks = pickle.load(open(test_masks, "rb"))

    return train_imgs, test_imgs, train_masks, test_masks


def load_raw_dataset():
    # path = '/network/rit/lab/ceashpc/chunpai/PycharmProjects/cnn_dvn/horse/'
    path = 'horse/'
    train_fn = 'train_raw.pdata'
    test_fn = 'test_raw.pdata'
    val_fn = 'val_raw.pdata'

    train_set = pickle.load(open(path + train_fn, "rb"))
    test_set = pickle.load(open(path + test_fn, "rb"))
    val_set = pickle.load(open(path + val_fn, "rb"))

    train_imgs = np.array([img for img in train_set['imgs']])
    val_imgs = np.array([img for img in val_set['imgs']])
    test_imgs = np.array([img for img in test_set['imgs']])

    train_masks = np.array([np.reshape(mask, (32, 32, 1)) for mask in train_set['segs']])
    val_masks = np.array([np.reshape(mask, (32, 32, 1)) for mask in val_set['segs']])
    test_masks = np.array([np.reshape(mask, (32, 32, 1)) for mask in test_set['segs']])
    print len(train_masks), len(val_masks), len(test_masks)
    np.set_printoptions(threshold=np.nan, linewidth=10000)

    all_imgs = np.concatenate((train_imgs, val_imgs, test_imgs), axis=0)
    for i, img in enumerate(all_imgs):
        im = Image.fromarray(np.uint8(img))
        im.save(path + "imgs/" + str(i) + ".png", format="PNG")

    all_masks = np.concatenate((train_masks, val_masks, test_masks), axis=0)
    for i, img in enumerate(all_masks):
        img = np.reshape(img, (32, 32))
        img[img > 0] = 255
        im = Image.fromarray(np.uint8(img))
        im.save(path + "masks/" + str(i) + ".png", format="PNG")

    # mask_fn = 'labels_2.npy'

    # images = np.load(os.path.join(path, image_fn))
    # # masks = np.load(os.path.join(path, mask_fn))
    #
    # # binarize
    # masks = np.asarray(masks > 128, dtype=np.float32)
    # split_idx = 200
    # if tag == 'train':
    #     # first 200 images as train samples
    #     return images[:split_idx], masks[:split_idx]
    # else:
    #     # the rest 128 images as test samples
    #     return images[split_idx:], masks[split_idx:]
    return train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks


def data_augmentation(train_imgs, train_masks):
    """data augmentation only apply to training set,
    here we randomly crop the 32*32 image and corresponding mask to 24*24,
    """
    # path = '/network/rit/lab/ceashpc/chunpai/PycharmProjects/cnn_dvn/horse/'
    path = 'horse/'
    train_imgs_cropped, train_masks_cropped = [], []
    for i, (img, mask) in enumerate(zip(train_imgs, train_masks)):
        x_offset = np.random.choice(range(8), 1)[0]
        y_offset = np.random.choice(range(8), 1)[0]
        img_cropped = img[x_offset: x_offset + 24, y_offset: y_offset + 24, :]
        mask_cropped = mask[x_offset: x_offset + 24, y_offset: y_offset + 24, :]
        train_imgs_cropped.append(img_cropped)
        train_masks_cropped.append(mask_cropped)

        # im = Image.fromarray(np.uint8(img_cropped))
        # im.save(path + "train_cropped_imgs/" + str(i) + ".png", format="PNG")
        #
        # mask = np.reshape(mask_cropped, (24, 24))
        # mask[mask > 0] = 255
        # ma = Image.fromarray(np.uint8(mask))
        # ma.save(path + "train_cropped_masks/" + str(i) + ".png", format="PNG")
    return np.array(train_imgs_cropped), np.array(train_masks_cropped)



def data_augmentation_train(train_imgs, train_masks):
    """data augmentation only apply to training set,
    """
    path = 'horse/'
    train_imgs_cropped, train_masks_cropped, train_values_cropped = [], [], []
    for i, (img, mask) in enumerate(zip(train_imgs, train_masks)):
        # single_imgs_cropped, single_masks_cropped = [], []
        for x_offset in range(1, 7, 1):
            for y_offset in range(1, 7, 1):
                img_cropped = img[x_offset: x_offset + 24, y_offset: y_offset + 24, :]
                mask_cropped = mask[x_offset: x_offset + 24, y_offset: y_offset + 24, :]
                train_imgs_cropped.append(img_cropped)
                train_masks_cropped.append(mask_cropped)
                train_values_cropped.append(1.0)

    for i, (img, true_mask) in enumerate(zip(train_imgs_cropped, train_masks_cropped)):
        for case in range(10):
            train_imgs_cropped.append(img)
            random_mask = np.random.uniform(0, 1, size=np.array(true_mask).shape)
            train_masks_cropped.append(random_mask)
            value = iou(np.array(true_mask), random_mask)
            train_values_cropped.append(value)
            print("value {}".format(value))

        for case in range(10):
            train_imgs_cropped.append(img)
            random_mask = true_mask * case / 10.0
            train_masks_cropped.append(random_mask)
            value = iou(np.array(true_mask), random_mask)
            train_values_cropped.append(value)
            print("value {}".format(value))

        for case in range(10):
            train_imgs_cropped.append(img)
            random_mask = np.random.randint(0, 2, size=np.array(true_mask).shape)
            train_masks_cropped.append(random_mask)
            value = iou(np.array(true_mask), random_mask)
            train_values_cropped.append(value)
            print("value {}".format(value))

        for case in range(1):
            train_imgs_cropped.append(img)
            random_mask = np.ones_like(true_mask)
            train_masks_cropped.append(random_mask)
            value = iou(np.array(true_mask), random_mask)
            train_values_cropped.append(value)
            print("value {}".format(value))

        for case in range(1):
            train_imgs_cropped.append(img)
            random_mask = 1 - true_mask
            train_masks_cropped.append(random_mask)
            value = iou(np.array(true_mask), random_mask)
            train_values_cropped.append(value)
            print("value {}".format(value))

    pickle.dump(np.array(train_imgs_cropped), open("data/train_imgs_augmented.pkl","wb"))
    pickle.dump(np.array(train_masks_cropped), open("data/train_masks_augmented.pkl","wb"))
    pickle.dump(np.array(train_values_cropped), open("data/train_values_augmented.pkl","wb"))
    # return np.array(train_imgs_cropped), np.array(train_masks_cropped), np.array(train_values_cropped)


def data_augmentation_test(test_imgs, test_masks):
    """data augmentation only apply to training set,
    here we randomly crop the 32*32 image and corresponding mask to 24*24,
    """
    # path = '/network/rit/lab/ceashpc/chunpai/PycharmProjects/cnn_dvn/horse/'
    path = 'horse/'
    test_imgs_cropped, test_masks_cropped = [], []
    for i, (img, mask) in enumerate(zip(test_imgs, test_masks)):
        # single_imgs_cropped, single_masks_cropped = [], []
        for x_offset in range(1, 7, 1):
            for y_offset in range(1, 7, 1):
                img_cropped = img[x_offset: x_offset + 24, y_offset: y_offset + 24, :]
                mask_cropped = mask[x_offset: x_offset + 24, y_offset: y_offset + 24, :]
                # single_imgs_cropped.append(img_cropped)
                # single_masks_cropped.append(mask_cropped)
                test_imgs_cropped.append(img_cropped)
                test_masks_cropped.append(mask_cropped)

        # im = Image.fromarray(np.uint8(img_cropped))
        # im.save(path + "train_cropped_imgs/" + str(i) + ".png", format="PNG")
        #
        # mask = np.reshape(mask_cropped, (24, 24))
        # mask[mask > 0] = 255
        # ma = Image.fromarray(np.uint8(mask))
        # ma.save(path + "train_cropped_masks/" + str(i) + ".png", format="PNG")
    return np.array(test_imgs_cropped), np.array(test_masks_cropped)


def view(masks):
    size, height, width, label_dim = masks.shape
    for i in range(size):
        print("----------------image:{}----------------".format(i))
        for j in range(height):
            for k in range(width):
                print int(masks[i][j][k][1]),
            print("")

def print_mask(mask):
    height, width, label_dim = mask.shape
    for j in range(height):
        for k in range(width):
            print int(mask[j][k][0]),
        print("")

def one_hot_encoding(mask):
    size, height, width, label_dim = mask.shape
    one_hot_mask = np.zeros([size, height, width, 2])
    for num in range(size):
        for x in range(height):
            for y in range(width):
                if mask[num, x, y] == 0:
                    one_hot_mask[num, x, y] = [1, 0]
                elif mask[num, x, y] == 1:
                    one_hot_mask[num, x, y] = [0, 1]
    # print one_hot_mask[0, :, :, 1]
    return one_hot_mask


def batch_iter(features, labels, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    features = np.array(features)
    labels = np.array(labels)

    data_size = len(features)
    num_batches_per_epoch = int((len(features) - 1) / batch_size) + 1
    # print("number of batches per epoch: {}".format(num_batches_per_epoch))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_features = features[shuffle_indices]
            shuffled_labels = labels[shuffle_indices]
        else:
            shuffled_features = features
            shuffled_labels = labels
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_features[start_index:end_index], shuffled_labels[start_index:end_index]


def softmax(input):
    # print("shape of input", input.shape)
    output = np.zeros_like(input)
    input = np.exp(input)
    sum = np.sum(input, axis=3) + 10e-10
    # print("shape of sum",sum.shape)
    for i in range(2):
        s = input[:, :, :, i]
        # print("s shape", s.shape)
        s /= sum
        output[:, :, :, i] = s
    output = np.array(output)
    # print("shape of output", output.shape)
    return output


def iou_one_hot(generated_labels, gt_labels):
    """the generalized intersect over union for one_hot_labels"""
    print("---------------------------------------------------")
    print("generated labels shape", generated_labels.shape)
    print("ground truth labels shape", gt_labels.shape)
    combo = np.concatenate((generated_labels, gt_labels), axis=0)
    print("combo shape", combo.shape)
    intersect = np.sum(np.min(combo, axis=0), axis=(0, 1))
    print("intersect", intersect)
    union = np.sum(np.max(combo, axis=0), axis=(0, 1))
    print("union", union)
    generated_value = intersect * 1.0 / union
    print("generated value shape", generated_value.shape)
    return generated_value


def iou(generated_labels, gt_labels):
    """the generalized intersect over union for one_hot_labels"""
    # print("---------------------------------------")
    # print("generated labels shape", generated_labels.shape)
    # print("ground truth labels shape", gt_labels.shape)
    # combo = np.concatenate((generated_labels, gt_labels), axis=0)
    # print("combo shape", combo.shape)
    combo = np.array([generated_labels, gt_labels])
    # print("combo shape", combo.shape)
    intersect = np.sum(np.min(combo, axis=0))
    # print("intersect", intersect)
    union = np.sum(np.max(combo, axis=0))
    # print("union", union)
    generated_value = intersect * 1.0 / union
    # print("generated value shape", generated_value.shape)
    return generated_value



def full_resolution(pred_labels, gt_labels_full):
    """fill the 24 by 24 crops back into 32 by 32 matrix, and compute the iou score"""
    pred_labels_full = np.zeros((32, 32, 1))
    count_full = np.zeros((32, 32, 1)) + 1e-10
    # print('gt_labels_full shape', gt_labels_full.shape)
    # print("pred labels shape", pred_labels.shape)
    count = 0
    for x_offset in range(1, 7, 1):
        for y_offset in range(1, 7, 1):
            pred_labels_full[x_offset: x_offset + 24, y_offset: y_offset + 24, :] += pred_labels[count]
            count_full[x_offset: x_offset + 24, y_offset: y_offset + 24, :] += 1.0
            count += 1
    # print("pred_labels_full shape", pred_labels_full.shape)
    # print(pred_labels_full[:, :, 0])
    pred_labels_full = np.divide(pred_labels_full, count_full)
    pred_labels_full[pred_labels_full >= 0.5] = 1.0
    pred_labels_full[pred_labels_full < 0.5] = 0.0
    # print("pred labels full", pred_labels_full[:, :, 0])
    # print("gt labels full", gt_labels_full[:, :, 0])
    foreground_value = iou(np.expand_dims(pred_labels_full, axis=0), np.expand_dims(gt_labels_full, axis=0))

    pred_labels_full = 1.0 - pred_labels_full
    gt_labels_full = 1.0 - gt_labels_full
    # print("pred labels full", pred_labels_full[:, :, 0])
    # print("gt labels full", gt_labels_full[:, :, 0])
    background_value = iou(np.expand_dims(pred_labels_full, axis=0), np.expand_dims(gt_labels_full, axis=0))
    # print('value', value)
    return foreground_value, background_value


def recover_full_resolution(pred_labels, gt_labels_full):
    """fill the 24 by 24 crops back into 32 by 32 matrix, and compute the iou score"""
    pred_labels_full = np.zeros((32, 32, 1))
    count_full = np.zeros((32, 32, 1)) + 1e-10
    # print('gt_labels_full shape', gt_labels_full.shape)
    # print("pred labels shape", pred_labels.shape)
    count = 0
    for x_offset in range(1, 7, 1):
        for y_offset in range(1, 7, 1):
            pred_labels_full[x_offset: x_offset + 24, y_offset: y_offset + 24, :] += pred_labels[count]
            count_full[x_offset: x_offset + 24, y_offset: y_offset + 24, :] += 1.0
            count += 1
    # print("pred_labels_full shape", pred_labels_full.shape)
    # print(pred_labels_full[:, :, 0])
    pred_labels_full = np.divide(pred_labels_full, count_full)
    pred_labels_full[pred_labels_full > 0.5] = 1.0
    pred_labels_full[pred_labels_full <= 0.5] = 0.0
    return pred_labels_full

if __name__ == '__main__':
    # train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks = load_pure_dataset()
    # data_augmentation_train(train_imgs, train_masks)

    load_raw_dataset()