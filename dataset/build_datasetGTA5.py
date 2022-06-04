import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data

class GTA5DataSet(data.Dataset):
    def __init__(self, root_path, target_folder=None, transformation=None, crop_size=(1024,512), mean=(128, 128, 128), ignore_label=255):
      self.root_path = root_path
      self.ignore_label = ignore_label
      self.crop_size = crop_size
      self.mean = mean
      self.files = []
      self.transformation = transformation
      self.target_folder = target_folder
      self.filestarget = []

      self.img_ids = [i_id.strip() for i_id in open(osp.join(self.root_path, "train.txt"))]

      self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

      for name in self.img_ids:
          img_file = osp.join(self.root_path, "images/%s" % name)
          label_file = osp.join(self.root_path, "labels/%s" % name)
          self.files.append({
              "img": img_file,
              "label": label_file
          })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        #Transformation
        if self.transformation is not None:
          im_lb = dict(im=image, lb=label)
          im_lb = self.transformation(im_lb)
          image, label = im_lb['im'], im_lb['lb']

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy()



