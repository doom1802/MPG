import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class GTA5DataSet(data.Dataset):
    #'./data/GTA' root_path
    def __init__(self, root_path, crop_size=(1024,512), mean=(128, 128, 128), std =(45,45,45), ignore_label=255):
      #ignore label=255 indica la label nulla
      self.root_path = root_path
      self.ignore_label = ignore_label
      self.crop_size = crop_size
      self.mean = mean
      self.files = []
      
      self.img_ids = [i_id.strip() for i_id in open(osp.join(self.root_path, "train.txt"))]

      self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

      # for split in ["train", "trainval", "val"]:
      for name in self.img_ids:
          img_file = osp.join(self.root_path, "images/%s" % name)
          label_file = osp.join(self.root_path, "labels/%s" % name)
          self.files.append({
              "img": img_file,
              "label": label_file
          })

      self.trans_train = Compose([
          HorizontalFlip(),
          RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
          RandomCrop(crop_size)
          #,GaussianBlur()
          ])

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy()


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BICUBIC)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )

class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )

class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BICUBIC),
                    lb = lb.resize((w, h), Image.NEAREST),
                )

class GaussianBlur(object):
    def __init__(self, *args, **kwargs):
        self.kernel = [(5,5), (7,7), (9,9)]
        self.sigma = np.linspace(0.1, 5, 5)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']

        i = int(random.random()*10) % 3
        j = int(random.random()*10) % 5
        im = T.functional.gaussian_blur(im, kernel_size=self.kernel[i],  sigma=self.sigma[j])
        return dict(im = im, lb = lb)

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb