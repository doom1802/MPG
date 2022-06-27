from utils.utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
import torch
import numpy as np

def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad(): 
        model.eval() 
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label, _) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            # model: BiseNet(args.num_classes, args.context_path)
            predict, _, _, _, _ = model(data)
            predict = predict.squeeze() 
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
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

