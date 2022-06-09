import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.build_datasetcityscapes import cityscapesDataSet
import torch
from utils.utils import rgb_label


def create_pseudo_labels(model, args, batch_size):

    if not os.path.exists(args.pseudo_path  + "/pseudolabels_3output" ):
      os.makedirs(args.pseudo_path + "/pseudolabels_3output")

    if not os.path.exists(args.pseudo_path + "/pseudolabels_rgb_3output"):
      os.makedirs(args.pseudo_path + "/pseudolabels_rgb_3output")
    
    model.eval()
    model.cuda()   

    dataset_train_Cityscapes = cityscapesDataSet(args.data_target)
    
    targetloader = DataLoader(
      dataset_train_Cityscapes,
      batch_size=1,
      shuffle=True,
      num_workers=1
    )

    target_train_loader_it = iter(targetloader)
    index = 0
    for i_iter in range(len(targetloader)):

        target_images, _, target_names = next(target_train_loader_it)

        image_name = []
        predicted_label = np.zeros((target_images.shape[0], 512, 1024))
        predicted_prob = np.zeros((target_images.shape[0], 512, 1024))

        for index, (image, name) in enumerate(zip(target_images, target_names)):
          if args.multi == 1:
            if image is not None:
              image = image.unsqueeze(0)
              output_sup, output_sup1, output_sup2, output_sup3, output_sup4 = model(image.cuda())

              output_sup = nn.functional.softmax(output_sup, dim=1)
              output_sup = nn.functional.upsample(output_sup, (512, 1024), mode='nearest').cpu().data[0].numpy()
              output_sup = output_sup.transpose(1,2,0)
              #prob_sup = np.argmax(output, axis=2)%19, np.max(output, axis=2)

              #output_sup1 = nn.functional.softmax(output_sup1, dim=1)
              #output_sup1 = nn.functional.upsample(output_sup1, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
              #output_sup1 = output_sup1.transpose(1,2,0)
              #prob_sup1 = np.argmax(output, axis=2)%19, np.max(output, axis=2)

              #output_sup2 = nn.functional.softmax(output_sup2, dim=1)
              #output_sup2 = nn.functional.upsample(output_sup2, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
              #output_sup2 = output_sup2.transpose(1,2,0)
              #prob_sup2 = np.argmax(output, axis=2)%19, np.max(output, axis=2)

              output_sup3 = nn.functional.softmax(output_sup3, dim=1)
              output_sup3 = nn.functional.upsample(output_sup3, (512, 1024), mode='nearest').cpu().data[0].numpy()
              output_sup3 = output_sup3.transpose(1,2,0)
              #prob_sup3 = np.argmax(output, axis=2)%19, np.max(output, axis=2)

              output_sup4 = nn.functional.softmax(output_sup4, dim=1)
              output_sup4 = nn.functional.upsample(output_sup4, (512, 1024), mode='nearest').cpu().data[0].numpy()
              output_sup4 = output_sup4.transpose(1,2,0)
              #prob_sup4 = np.argmax(output, axis=2)%19, np.max(output, axis=2)
              
              output = np.concatenate((output_sup, output_sup3, output_sup4), axis=2)
              label, prob = np.argmax(output, axis=2)%19, np.max(output, axis=2)

              # majority voting
              #output = np.concatenate((prob_sup, prob_sup3, prob_sup4), axis=2)
              #label, prob = np.argmax(output, axis=2)%19, np.max(output, axis=2)

              predicted_label[index] = label.copy()
              predicted_prob[index] = prob.copy()
              image_name.append(name[0])
          else:
            if image is not None:
              image = image.unsqueeze(0)
              predictions = model(image.cuda())

              output = nn.functional.softmax(predictions[0], dim=1)
              output = nn.functional.upsample(output, (512, 1024), mode='nearest').cpu().data[0].numpy()
              output = output.transpose(1,2,0)
      
              label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
              predicted_label[index] = label.copy()
              predicted_prob[index] = prob.copy()
              image_name.append(name)

        thres = []
        for i in range(19):
            x = predicted_prob[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x) * 0.5))])
        thres = np.array(thres)
        thres[thres > 0.9] = 0.9

        for idx in range(target_images.shape[0]):
            name = image_name[idx]
            name = name.replace("leftImg8bit", "gtFine_labelIds")
            label = predicted_label[idx]
            prob = predicted_prob[idx]

            # creare la label di conseguenza
            for i in range(19):
                label[(prob < thres[i]) * (label == i)] = 255

            output = np.asarray(label, dtype=np.uint8)
            rgb_image = rgb_label(output)
            output = Image.fromarray(output)

            output.save('%s/%s' % (args.pseudo_path + "/pseudolabels_output", name))
            rgb_image.save('%s/%s' % (args.pseudo_path + "/pseudolabels_rgb_output", name))
