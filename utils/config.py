import argparse
import numpy as np

def get_args(params):
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
  parser.add_argument('--pseudo-path', type=str, default='', help='path of pseudo data')
  parser.add_argument("--multi", type=int, default=0, help='ok')
  
  parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
  parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
  parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
  parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
  parser.add_argument('--use_pretrained_model', type=int, default=0, help='path to pretrained model')
  parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
  parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
  parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
  parser.add_argument('--input-size', type=str, default='1280,720', help='optimizer, support rmsprop, sgd, adam')
  parser.add_argument('--input-size-target', type=str, default='1024,512', help='loss function, dice or crossentropy')
  
  parser.add_argument('--create-pseudolabels', type=int, default=50, help='loss function, dice or crossentropy')
  parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2.')
  
  parser.add_argument("--power", type=float, default=0.9,
                      help="Decay parameter to compute the learning rate.")
  parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                      help="Base learning rate for discriminator.")
  parser.add_argument("--iter-size", type=int, default=1,
                      help="Accumulate gradients for ITER_SIZE iterations.")
  parser.add_argument("--lambda-seg", type=float, default=0.1,
                      help="lambda_seg.")
  parser.add_argument("--lambda-adv-target", type=float, default=0.001,
                      help="lambda_adv for adversarial training.")
  parser.add_argument("--lambda-adv-target1", type=float, default=0.002,
                        help="lambda_adv for adversarial training.")
  parser.add_argument("--lambda-adv-target2", type=float, default=0.004,
                        help="lambda_adv for adversarial training.")
  parser.add_argument("--transformation_on_source", type=str, default=None, help="Could be LAB or FDA. None means no transformation.")
  parser.add_argument('--ligth_weigth', type=str, default=None, help="Could be LAB or FDA. None means no transformation.")
  parser.add_argument("--ssl", type=str, default=None, help="enable self supervised learning")
  parser.add_argument("--update-pseudo-labels", type=int, default=1, help="How many epochs pseudo labels should be updated.")
  parser.add_argument('--checkpoint_name', type=str, default='', help='models are saved here')
        
  args = parser.parse_args(params)

  img_mean = np.array((73.158359210711552, 82.908917542625858, 72.392398761941593), dtype=np.float32)

  w, h = map(int, args.input_size.split(','))    #source
  input_size = (w, h)

  w, h = map(int, args.input_size_target.split(','))   #target
  input_size_target = (w, h)

  return args, input_size, input_size_target, img_mean
