from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
from torch.autograd import Variable
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils.utils import poly_lr_scheduler, load_da_model, save_da_model
import torch.cuda.amp as amp
from model.build_discriminator import FCDiscriminator, FCDiscriminatorLight
from dataset.build_datasetGTA5 import GTA5DataSet
from dataset.build_datasetcityscapes import cityscapesDataSet
from utils.config import get_args
from validation import val
from utils.SSL import create_pseudo_labels


def train(args, model, optimizer, trainloader, targetloader, model_D, optimizer_D, targetloader_val, mean, crop_size): #passo gli args, il modello, l'optimizer e il dataloader
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()
    discriminator_scaler = amp.GradScaler()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    max_miou = 0 
    step = 0
    source_label = 0
    target_label = 1
    for epoch in range(args.num_epochs):

        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs, power=args.power)
        discriminator_lr = poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs, power=args.power)
        
        model.train()
        model_D.train()

        tq = tqdm(total=len(trainloader) * args.batch_size)
        tq.set_description('epoch %d, lr %f'% (epoch , lr))
         
        loss_record_source = []
        loss_record_target = []
        loss_D_record = []

        for batch_train, batch_target in zip(enumerate(trainloader), enumerate(targetloader)):
            
            _, (data_train, label_train) = batch_train

            if args.ssl == 0:
              _, (data_target, _, _) = batch_target
            else:
              _, (data_target, label_target, _) = batch_target

            optimizer.zero_grad()
            optimizer_D.zero_grad()
            
          # train Segmentation network

            for param in model_D.parameters():
                param.requires_grad = False
            
            # train with source
            data_train = data_train.cuda()
            label_train = label_train.long().cuda()
            
            with amp.autocast():
                output, output_sup1, output_sup2, output_sup3, output_sup4 = model(data_train) 
                loss1 = loss_func(output, label_train)
                loss2 = 0.0#loss_func(output_sup1, label_train)
                loss3 = 0.0#loss_func(output_sup2, label_train)
                loss4 = loss_func(output_sup3, label_train)
                loss5 = loss_func(output_sup4, label_train)
                loss_seg_source = loss1 + loss2 + loss3 + loss4 + loss5

            scaler.scale(loss_seg_source).backward()
            
            # train with target
            data_target = data_target.cuda()  

            if args.ssl == 1:                           
             label_target = label_target.long().cuda()         
            
            with amp.autocast():
                output_t, output_sup1_t, output_sup2_t, output_sup3_t, output_sup4_t = model(data_target)
                if args.ssl == 1:
                  loss1t = loss_func(output_t, label_target)
                  loss2t = 0.0 #loss_func(output_sup1_t, label_target)
                  loss3t = 0.0 #loss_func(output_sup2_t, label_target)
                  loss4t = loss_func(output_sup3_t, label_target)
                  loss5t = loss_func(output_sup4_t, label_target)
                  loss_seg_target = loss1t + loss2t + loss3t + loss4t + loss5t
                else:
                  loss_seg_target = 0.0

                D_out = model_D(F.softmax(output_t, dim=1))

                loss_adv_target = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
                loss_target = args.lambda_adv_target * loss_adv_target + loss_seg_target

            scaler.scale(loss_target).backward()
            
            # train Discriminator network

            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            with amp.autocast():
              output = output.detach()
              D_out = model_D(F.softmax(output, dim=1))
              loss_D_source = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
 
            # train with target
            with amp.autocast():
              output_t = output_t.detach()
              D_out = model_D(F.softmax(output_t, dim=1))
              loss_D_target = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())
              
              loss_D = loss_D_source/2 + loss_D_target/2

            discriminator_scaler.scale(loss_D).backward()

            discriminator_scaler.step(optimizer_D)
            scaler.step(optimizer)

            discriminator_scaler.update()
            scaler.update()
            
            tq.update(args.batch_size)
            
            tq.set_postfix(loss_seg_source='%.6f' % loss_seg_source, loss_target='%.6f' % loss_target, loss_D='%.6f' % loss_D)
            step += 1
            writer.add_scalar('loss_seg_source_step', loss_seg_source, step)
            writer.add_scalar('loss_target_step', loss_target, step)
            writer.add_scalar('loss_D_step', loss_D, step)
            
            loss_record_source.append(loss_seg_source.item())
            loss_record_target.append(loss_target.item())
            loss_D_record.append(loss_D.item())

        tq.close()

        loss_train_mean_source = np.mean(loss_record_source)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean_source), epoch)
        print('loss for train source : %f' % (loss_train_mean_source))

        loss_train_mean_target = np.mean(loss_record_target)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean_target), epoch)
        print('loss for train target : %f' % (loss_train_mean_target))

        loss_D_mean = np.mean(loss_D_record)
        writer.add_scalar('epoch/loss_', float(loss_D_mean), epoch)
        print('loss for discriminator : %f' % (loss_D_mean))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            save_da_model(args, model, model_D, optimizer, optimizer_D, epoch)

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, targetloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                save_da_model(args, model, model_D, optimizer, optimizer_D, epoch, "best")
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
            

def main(params):

  args, input_size, input_size_target, img_mean = get_args(params)

  print("Actual parameters")
  print("ssl: " + str(args.ssl))
  print("use_pretrained_model: " + str(args.use_pretrained_model))
  print("ligth_weigth: " + str(args.ligth_weigth))
  print("checkpoint_name_load: " + args.checkpoint_name_load)
  print("checkpoint_name_save: " + args.checkpoint_name_save)


  # create dataset and dataloader   
  
  dataset_train_GTA = GTA5DataSet(args.data_source, target_folder = args.data_target, mean = img_mean, crop_size = input_size)

  trainloader = DataLoader(
      dataset_train_GTA,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers
  )
  if args.ssl == 1:
    if args.multi == 2:
      pseudo_path = "/pseudolabels_2output" 
      args.checkpoint_name_save = args.checkpoint_name_save.replace(".pth", "_2output.pth")
    elif args.multi == 3:
      pseudo_path = "/pseudolabels_3output" 
      args.checkpoint_name_save = args.checkpoint_name_save.replace(".pth", "_3output.pth")
    else:
      pseudo_path = "/pseudolabels"
      args.checkpoint_name_save = args.checkpoint_name_save.replace(".pth", "_noMulti.pth")

  if args.ssl == 0:
    dataset_train_Cityscapes= cityscapesDataSet(args.data_target, mean=img_mean, mode='train', crop_size=input_size_target)

    targetloader = DataLoader(
        dataset_train_Cityscapes,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
  else:
    dataset_train_Cityscapes= cityscapesDataSet(args.data_target, pseudo_path = args.pseudo_path + pseudo_path, mean = img_mean, mode='train', crop_size=input_size_target, ssl = True)

    targetloader = DataLoader(
        dataset_train_Cityscapes,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

  dataset_val = cityscapesDataSet(args.data_target, mean=img_mean, mode='val', crop_size=input_size_target)

  targetloader_val = DataLoader(
      dataset_val,
      batch_size=1,
      shuffle=True,
      num_workers=args.num_workers
  )

  # build model
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


  model = BiSeNet(args.num_classes, args.context_path)
  if torch.cuda.is_available() and args.use_gpu:
      model = torch.nn.DataParallel(model).cuda()

  # build optimizer
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)

  if args.ligth_weigth == 0: 
    model_D = FCDiscriminator(num_classes=args.num_classes)
  else:
    model_D = FCDiscriminatorLight(num_classes=args.num_classes)
  
  if torch.cuda.is_available() and args.use_gpu:
    model_D = torch.nn.DataParallel(model_D).cuda()

  optimizer_D = torch.optim.Adam(model_D.parameters(), args.learning_rate_D, betas=(0.9, 0.99))
  
  if args.use_pretrained_model == 1:
      model, model_D, optimizer, optimizer_D, epoch_start = load_da_model(args, model, model_D, optimizer, optimizer_D)
 
  if args.use_pretrained_model == 1:
    val(args, model, targetloader_val)
    if args.ssl == 1:
      create_pseudo_labels(model, args, batch_size=1)
  else:
    # train
    train(args, model, optimizer, trainloader, targetloader, model_D, optimizer_D, targetloader_val, img_mean, input_size_target)
    # final test
    val(args, model, targetloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--num_workers', '4',
        '--num_classes', '19',
        '--validation_step', '5',
        '--checkpoint_step', '5',
        '--cuda', '0',
        '--batch_size', '4',
        '--data-source', './data/GTA5',
        '--data-target', './data/Cityscapes',
        '--pseudo-path', './data/Pseudo_Cityscapes',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--save_model_path', './checkpoints',
        '--checkpoint_name_load', 'model_unsupervised_best.pth',
        '--checkpoint_name_save', 'model_unsupervised_3output.pth',
        '--multi', '3', 
        '--ssl', '1',
        '--use_pretrained_model', '0',
        '--ligth_weigth', '0',
    ]
    main(params)