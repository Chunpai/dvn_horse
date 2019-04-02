from scipy.io import loadmat
from PIL import Image
import numpy as np
import pickle

x = loadmat('horse.mat')
print len(x['imgs'])
print len(x['segs'])
imgs_dict = {}
masks_dict = {}
for i, img in enumerate(x['imgs']):
    img = np.array(img[0])
    imgs_dict[i] = img
for i, mask in enumerate(x['segs']):
    # print len(mask)
    mask = np.reshape(mask, (32, 32))
    mask = np.flip(mask, 0)
    mask = np.rot90(mask, k=1, axes=(1,0))
    masks_dict[i] = mask
print x.keys()


y = loadmat('weizmann_32_32_trainval.mat')
print len(y["xtrain"])
train_imgs = []
train_masks = []
for i, mask in enumerate(y['xtrain']):
    mask = np.reshape(mask, (32, 32))
    mask = np.flip(mask, 0)
    mask = np.rot90(mask, k=1, axes=(1,0))
    for j, ma in masks_dict.items():
        if np.array_equal(mask, ma):
            print True
            mask[mask > 0] = 255
            im = Image.fromarray(np.uint8(mask))
            im.save("../train/{}mask.png".format(i),format("PNG"))
            mask[mask > 0] = 1
            train_masks.append(list(np.expand_dims(mask, axis=2)))

            img = imgs_dict[j]
            train_imgs.append(list(img))
            im = Image.fromarray(np.uint8(img))
            im.save("../train/{}img.png".format(i),format("PNG"))
pickle.dump(np.array(train_imgs), open("../train/train_imgs.pkl","wb"))
pickle.dump(np.array(train_masks), open("../train/train_masks.pkl","wb"))


test_masks = []
test_imgs = []
for i, mask in enumerate(y['xval']):
    mask = np.reshape(mask, (32, 32))
    mask = np.flip(mask, 0)
    mask = np.rot90(mask, k=1, axes=(1,0))
    for j, ma in masks_dict.items():
        if np.array_equal(mask, ma):
            print True
            mask[mask > 0] = 255
            im = Image.fromarray(np.uint8(mask))
            im.save("../test/{}mask.png".format(i),format("PNG"))
            mask[mask > 0] = 1
            test_masks.append(list(np.expand_dims(mask, axis=2)))

            img = imgs_dict[j]
            test_imgs.append(list(img))
            im = Image.fromarray(np.uint8(img))
            im.save("../test/{}img.png".format(i),format("PNG"))
pickle.dump(np.array(test_imgs), open("../test/test_imgs.pkl","wb"))
pickle.dump(np.array(test_masks), open("../test/test_masks.pkl","wb"))

# # print y
# print y['utrain']
# # img = np.array(y['xtrain'][0])
# # img = np.reshape(img, (32, 32))
# # img[img > 0] = 255
# # im = Image.fromarray(np.uint8(img))
# # im.save("horse23.png",format("PNG"))
print y.keys()
print y['idx']
