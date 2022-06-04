import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class cityscapesDataSet(data.Dataset):
    def __init__(self, root_path, pseudo_path=None, crop_size=(1024,512), mean=(73.158359210711552, 82.908917542625858, 72.392398761941593), ignore_label=255, transformation=None, ssl = None, mode="train"):
      self.mode = mode
      self.root_path = root_path
      self.crop_size = crop_size
      self.ignore_label = ignore_label
      self.mean = mean
      self.transformation=transformation 
      self.pseudo_path = pseudo_path
      self.files = []
      self.ssl = ssl

      self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
      
      self.img_ids = [i_id.split("/")[1].strip() for i_id in open(osp.join(self.root_path, mode+".txt"))]
      
      for name in self.img_ids:
          img_file = osp.join(self.root_path, "images", name)
          if (self.ssl is not None):
            label_file = osp.join(self.pseudo_path, name.replace("leftImg8bit", "gtFine_labelIds"))
          else:
            label_file = osp.join(self.root_path, "labels", name.replace("leftImg8bit", "gtFine_labelIds"))

          self.files.append({
              "img": img_file,
              "label": label_file,
              "name": name,
          })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
      datafiles = self.files[index]

      image = Image.open(datafiles["img"]).convert('RGB')
      label = Image.open(datafiles["label"])
      name = datafiles['name']

      image = image.resize(self.crop_size, Image.BICUBIC)
      label = label.resize(self.crop_size, Image.NEAREST)

      if self.mode == 'train' and self.transformation is not None:
        im_lb = dict(im=image, lb=label)
        im_lb = self.transformation(im_lb)
        image, label = im_lb['im'], im_lb['lb']

      image = np.asarray(image, np.float32)
      label = np.asarray(label, np.float32)

      
      if self.ssl is None:
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
      else:
        label_copy = label

      image = image[:, :, ::-1] 
      image -= self.mean
      image = image.transpose((2, 0, 1))

      return image.copy(), label_copy.copy(), name
