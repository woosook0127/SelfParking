import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './dataset'
input_dir = './dataset/images'
label_dir = './dataset/annotations/trimaps'

name_input = sorted([os.path.join(input_dir, fname)
                          for fname in os.listdir(input_dir)
                             if fname.endswith('.jpg')])
name_label = sorted([os.path.join(label_dir, fname)
                          for fname in os.listdir(label_dir)
                             if fname.endswith('.png') and not fname.startswith('.')])

print(len(name_input), len(name_label))

## 데이터 나누기 (train, test, val)
import random

test_samples = 1000
val_samples = 500
train_samples = len(name_input) - test_samples - val_samples

random.Random(88).shuffle(name_input)
random.Random(88).shuffle(name_label)

train_input_img_paths = name_input[:-(test_samples + val_samples)]
train_label_img_paths = name_label[:-(test_samples + val_samples)]

test_input_img_paths = name_input[train_samples:-val_samples]
test_label_img_paths = name_label[train_samples:-val_samples]

val_input_img_paths = name_input[-val_samples:]
val_label_img_paths = name_label[-val_samples:]

print(len(train_label_img_paths), len(test_label_img_paths), len(val_label_img_paths))

##
dir_save_train = os.path.join(dir_data, 'train')
dir_save_test = os.path.join(dir_data, 'test')
dir_save_val = os.path.join(dir_data, 'val')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
##
for i in range(len(train_label_img_paths)):
    label_ = np.asarray(Image.open(train_label_img_paths[i]).resize((300,300)))
    input_ = np.asarray(Image.open(train_input_img_paths[i]).resize((300,300)))

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

for i in range(len(test_label_img_paths)):
    label_ = np.asarray(Image.open(test_label_img_paths[i]).resize((300,300)))
    input_ = np.asarray(Image.open(test_input_img_paths[i]).resize((300,300)))

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

for i in range(len(val_label_img_paths)):
    label_ = np.asarray(Image.open(val_label_img_paths[i]).resize((300,300)))
    input_ = np.asarray(Image.open(val_input_img_paths[i]).resize((300,300)))

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

##
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()
##

