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
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
import torch.cuda.amp as amp
from model.discriminator_light import FCDiscriminator
from dataset.build_datasetGTA5 import GTA5DataSet
from dataset.build_datasetcityscapes import cityscapesDataSet

IMG_MEAN = np.array((73.158359210711552, 82.908917542625858, 72.392398761941593), dtype=np.float32)


def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad(): 
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            # model: BiseNet(args.num_classes, args.context_path)
            predict = model(data).squeeze() #squeeze toglie le dimensioni pari ad 1 
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def train(args, model, optimizer, trainloader, targetloader, model_D, model_D1, model_D2, optimizer_D, optimizer_D1, optimizer_D2, bce_loss, targetloader_val): #passo gli args, il modello, l'optimizer e il dataloader
#optimizer: procedura attraverso la quale si aggiornano i pesi in direzione del gradiente 
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
#TensorBoard is a web interface that reads data from a file and displays it. To make this easy for us, PyTorch has a utility class called SummaryWriter. 
#The SummaryWriter class is your main entry to log data for visualization by TensorBoard.
    scaler = amp.GradScaler() #gradinet init

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0 
    step = 0
    source_label = 0#da capire
    target_label = 1
    for epoch in range(args.num_epochs):
        
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr1 = poly_lr_scheduler(optimizer_D, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr2 = poly_lr_scheduler(optimizer_D1, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr3 = poly_lr_scheduler(optimizer_D2, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        
        model.train()
        model_D.train()
        model_D1.train()
        model_D2.train()
        tq = tqdm(total=len(trainloader) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

        for batch_train, batch_target in zip(enumerate(trainloader), enumerate(targetloader)):
            
            _, (data_train, label_train) = batch_train
            _, (data_target, label_target) = batch_target

            optimizer.zero_grad()
            optimizer_D.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()

            # train Segmentation network

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            for param in model_D1.parameters():
                param.requires_grad = False
            
            for param in model_D2.parameters():
                param.requires_grad = False
            
            # train with source
            data_train = data_train.cuda()
            label_train = label_train.long().cuda()
            
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data_train) 
                loss1 = loss_func(output, label_train)
                loss2 = loss_func(output_sup1, label_train)
                loss3 = loss_func(output_sup2, label_train)
                loss = loss1 + loss2 + loss3
            scaler.scale(loss).backward()
            

            # train with target
            data_target = data_target.cuda()
            
            with amp.autocast():
                output_t, output_sup1_t, output_sup2_t = model(data_target) 
                D_out = model_D(output_t)
                D_out1 = model_D1(output_sup1_t)
                D_out2 = model_D2(output_sup2_t)

                loss_adv_target = bce_loss(D_out,
                                       Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

                loss_adv_target1 = bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
                                          
                loss_adv_target2 = bce_loss(D_out2,
                                       Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

                loss = args.lambda_adv_target * loss_adv_target + args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            
            scaler.scale(loss).backward()

            # train Discriminator network

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True
            
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            with amp.autocast():
              output = output.detach()
              output_sup1 = output_sup1.detach()
              output_sup2 = output_sup2.detach()

              D_out = model_D(output)
              D_out1 = model_D1(output_sup1)
              D_out2 = model_D2(output_sup2)

              loss_D_s = bce_loss(D_out,
                                        Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

              loss_D1_s = bce_loss(D_out1,
                                      Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
                                        
              loss_D2_s = bce_loss(D_out2,
                                      Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())


            scaler.scale(loss_D_s).backward()
            scaler.scale(loss_D1_s).backward()
            scaler.scale(loss_D2_s).backward()

            # train with target
            
            with amp.autocast():
              output_t = output_t.detach()
              output_sup1_t = output_sup1_t.detach()
              output_sup2_t = output_sup2_t.detach()

              D_out = model_D(output_t)
              D_out1 = model_D1(output_sup1_t)
              D_out2 = model_D2(output_sup2_t)

              loss_D_s = bce_loss(D_out,
                                        Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())

              loss_D1_s = bce_loss(D_out1,
                                      Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
                                        
              loss_D2_s = bce_loss(D_out2,
                                      Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())


            scaler.scale(loss_D_s).backward()
            scaler.scale(loss_D1_s).backward()
            scaler.scale(loss_D2_s).backward()
            

            scaler.step(optimizer)
            scaler.step(optimizer_D)
            scaler.step(optimizer_D1)
            scaler.step(optimizer_D2)
            scaler.update()
            
            tq.update(args.batch_size)
            
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            #writer.add_scalar('loss_step', loss_t_t, step)
            
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, targetloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data-source', type=str, default='', help='path of training data')
    parser.add_argument('--data-target', type=str, default='', help='path of training data')
    
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--input-size', type=str, default='1280,720', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--input-size-target', type=str, default='1024,512', help='loss function, dice or crossentropy')
    
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--iter-size", type=int, default=1,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--lambda-seg", type=float, default=0.1,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target", type=float, default=0.0004,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target1", type=float, default=0.0002,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=0.001,
                        help="lambda_adv for adversarial training.")
    
    args = parser.parse_args(params)


    w, h = map(int, args.input_size.split(','))    #source
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))   #target
    input_size_target = (w, h)

    # create dataset and dataloader             
    dataset_train_GTA = GTA5DataSet(args.data_source,mean=IMG_MEAN,crop_size=input_size)

    trainloader = DataLoader(
        dataset_train_GTA,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataset_train_Cityscapes= cityscapesDataSet(args.data_target,mean=IMG_MEAN, crop_size=input_size_target)

    targetloader = DataLoader(
        dataset_train_Cityscapes,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataset_val = cityscapesDataSet(args.data_target, mean=IMG_MEAN, mode='val', crop_size=input_size_target)

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
    optimizer_D = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer_D1 = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer_D2 = torch.optim.Adam(model.parameters(), args.learning_rate)
    
    model_D = FCDiscriminator(num_classes=args.num_classes)

    model_D.cuda()

    model_D1 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.cuda()

    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D2.cuda()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, trainloader, targetloader, model_D, model_D1, model_D2, optimizer_D, optimizer_D1, optimizer_D2, bce_loss, targetloader_val)
    # final test
    val(args, model, targetloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--data-source', './data/GTA5',
        '--data-target', './data/Cityscapes',
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '8',
        '--save_model_path', './checkpoints_18_sgd',
        '--context_path', 'resnet18',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',

    ]
    main(params)

