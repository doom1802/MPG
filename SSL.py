import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.build_datasetcityscapes import cityscapesDataSet
import torch

palette = [128, 64, 128,  # Road, 0
            244, 35, 232,  # Sidewalk, 1
            70, 70, 70,  # Building, 2
            102, 102, 156,  # Wall, 3
            190, 153, 153,  # Fence, 4
            153, 153, 153,  # pole, 5
            250, 170, 30,  # traffic light, 6
            220, 220, 0,  # traffic sign, 7
            107, 142, 35,  # vegetation, 8
            152, 251, 152,  # terrain, 9
            70, 130, 180,  # sky, 10
            220, 20, 60,  # person, 11
            255, 0, 0,  # rider, 12
            0, 0, 142,  # car, 13
            0, 0, 70,  # truck, 14
            0, 60, 100,  # bus, 15
            0, 80, 100,  # train, 16
            0, 0, 230,  # motor-bike, 17
            119, 11, 32]  # bike, 18]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def create_pseudo_labels(model, args, batch_size):

    if not os.path.exists(args.pseudo_path  + "/pseudolabels" ):
      os.makedirs(args.pseudo_path + "/pseudolabels")

    if not os.path.exists(args.pseudo_path + "/pseudolabels_rgb"):
      os.makedirs(args.pseudo_path + "/pseudolabels_rgb")
    
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

        for index, (timage, tname) in enumerate(zip(target_images, target_names)):
            if timage is not None:
                timage = timage.unsqueeze(0)

                predictions = model(timage.cuda())

                output = nn.functional.softmax(predictions[0], dim=1)

                output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()

                output = output.transpose(1,2,0)
        
                #timage = timage.unsqueeze(0)
                #predictions = model(timage.cuda())

                #target_prediction = predictions[0].unsqueeze(0)
                
                
                #output = torch.nn.functional.softmax(target_prediction, dim=1)
                #print(output.size())
                #output = torch.nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = output.transpose(1, 2, 0)
                #print(output.size())
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
                predicted_label[index] = label.copy()
                predicted_prob[index] = prob.copy()
                image_name.append(tname)

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
            label = predicted_label[idx]
            prob = predicted_prob[idx]

            # creare la label di conseguenza
            for i in range(19):
                label[(prob < thres[i]) * (label == i)] = 255

            output = np.asarray(label, dtype=np.uint8)
            mask_img = colorize_mask(output)
            output = Image.fromarray(output)

            output.save('%s/%s' % (args.pseudo_path + "/pseudolabels", name.replace("leftImg8bit", "gtFine_labelIds")))
            mask_img.save('%s/%s' % (args.pseudo_path + "/pseudolabels_rgb", name.replace("leftImg8bit", "gtFine_labelIds")))

       
'''
    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []

    for index, (image, label, name) in enumerate(targetloader):
      if args.multi == 1:
        if image is not None:
          output_sup, output_sup1, output_sup2, output_sup3, output_sup4 = model(Variable(image).cuda())

          output_sup = nn.functional.softmax(output_sup, dim=1)
          output_sup = nn.functional.upsample(output_sup, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
          output_sup = output_sup.transpose(1,2,0)
          #label_sup, prob_sup = np.argmax(output, axis=2)%19, np.max(output, axis=2)

          output_sup1 = nn.functional.softmax(output_sup1, dim=1)
          output_sup1 = nn.functional.upsample(output_sup1, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
          output_sup1 = output_sup1.transpose(1,2,0)
          #label_sup1, prob_sup1 = np.argmax(output, axis=2)%19, np.max(output, axis=2)

          output_sup2 = nn.functional.softmax(output_sup2, dim=1)
          output_sup2 = nn.functional.upsample(output_sup2, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
          output_sup2 = output_sup2.transpose(1,2,0)
          #label_sup2, prob_sup2 = np.argmax(output, axis=2)%19, np.max(output, axis=2)

          output_sup3 = nn.functional.softmax(output_sup3, dim=1)
          output_sup3 = nn.functional.upsample(output_sup3, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
          output_sup3 = output_sup3.transpose(1,2,0)
          #label_sup3, prob_sup3 = np.argmax(output, axis=2)%19, np.max(output, axis=2)

          output_sup4 = nn.functional.softmax(output_sup4, dim=1)
          output_sup4 = nn.functional.upsample(output_sup4, (512, 1024), mode='nearest', align_corners=True).cpu().data[0].numpy()
          output_sup4 = output_sup4.transpose(1,2,0)
          #label_sup4, prob_sup4 = np.argmax(output, axis=2)%19, np.max(output, axis=2)
          
          output = np.concatenate((output_sup, output_sup1, output_sup2, output_sup3, output_sup4), axis=2)
          label, prob = np.argmax(output, axis=2)%19, np.max(output, axis=2)

          #labels_conc = np.concatenate((label_sup,label_sup1, label_sup2,label_sup3, label_sup4), axis=2)
          #probs_conc = np.concatenate((prob_sup, prob_sup1, prob_sup2, prob_sup3, prob_sup4), axis=2)


          # majority voting
          
          predicted_label[index] = label.copy()
          predicted_prob[index] = prob.copy()
          image_name.append(name[0])
      else:
      if image is not None:
        output,_,_,_,_ = model(image.cuda())
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        print("prova SSL")
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        print(np.unique(label))
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])

         for index in range(len(targetloader)):
            name = image_name[index]
            label = predicted_label[index]
            prob = predicted_prob[index]
            
            print(np.unique(label))
            for i in range(19):
              label[(prob < thres[i]) * (label == i)] = 255
            print(np.unique(label))
            output = np.asarray(label, dtype=np.uint8)
            
            output = Image.fromarray(output)
            output.save('%s/%s' % (args.pseudo_path, name.replace("leftImg8bit", "gtFine_labelIds"))) 
'''


        
    