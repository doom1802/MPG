import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
from torch.autograd import Variable
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.utils import poly_lr_scheduler, save_model, load_model
import torch.cuda.amp as amp
from dataset.build_datasetcityscapes import cityscapesDataSet
from utils.config import get_args
from validation import val
from utils.augumentation import Compose, HorizontalFlip, RandomCrop, RandomScale

def train(args, model, optimizer, dataloader_train, dataloader_val): #passo gli args, il modello, l'optimizer e il dataloader
#optimizer: procedura attraverso la quale si aggiornano i pesi in direzione del gradiente 
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
#TensorBoard is a web interface that reads data from a file and displays it. To make this easy for us, PyTorch has a utility class called SummaryWriter. 
#The SummaryWriter class is your main entry to log data for visualization by TensorBoard.
    scaler = amp.GradScaler() #gradinet init

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0 
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train() 
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        
        for i, (data, label, _) in enumerate(dataloader_train):
            data = Variable(data).cuda()
            label = Variable(label).long().cuda()
            optimizer.zero_grad()
            
            with amp.autocast():
                output, _, _, output_sup1, output_sup2 = model(data) 
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            save_model(args, model, optimizer, epoch)

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                save_model(args, model, optimizer, epoch, "best")
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    args, _, input_size, img_mean = get_args(params)

    # create dataset and dataloader   
    transformation = Compose ([HorizontalFlip(), RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),RandomCrop(input_size)])  

    dataset_train = cityscapesDataSet(args.data_target, transformation = transformation, mean=img_mean, mode='train', crop_size=input_size)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataset_val = cityscapesDataSet(args.data_target, mean=img_mean, mode='val', crop_size=input_size)

    dataloader_val = DataLoader(
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

    if args.use_pretrained_model == 1:
      model, optimizer, epoch_start = load_model(args, model, optimizer)
 
  
    if args.use_pretrained_model == 1:
      val(args, model, dataloader_val)
    else:
      # train
      train(args, model, optimizer, dataloader_train, dataloader_val)
      # final test
      val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--data-target', './data/Cityscapes',
        '--num_workers', '8',
        '--num_classes', '19',
        '--save_model_path', './checkpoints',
        '--cuda', '0',
        '--batch_size', '8',
        '--checkpoint_name_load', 'model_resnet18_best.pth', #'model_unsupervisedSSL_3output.pth'
        '--checkpoint_name_save', 'model_resnet18.pth',
        '--context_path', 'resnet18',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        '--use_pretrained_model', '1',
    ]
    main(params)

