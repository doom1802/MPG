import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import numbers
import torchvision
import os

palette = [128, 64, 128,  # road, 0
            244, 35, 232,  # sidewalk, 1
            70, 70, 70,  # building, 2
            102, 102, 156,  # wall, 3
            190, 153, 153,  # fence, 4
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
            119, 11, 32]  # bike, 18

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def rgb_label(label):

    new_mask = Image.fromarray(label.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask



def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def adjust_learning_rate_D(optimizer, epoch, args):
    lr = lr_poly(args.learning_rate_D, epoch, args.num_epochs, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
	# if iter % lr_decay_iter or iter > max_iter:
	# 	return optimizer

	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr
	# return lr

def get_label_info(csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	label = {}
	for iter, row in ann.iterrows():
		label_name = row['name']
		r = row['r']
		g = row['g']
		b = row['b']
		class_11 = row['class_11']
		label[label_name] = [int(r), int(g), int(b), class_11]
	return label

def one_hot_it(label, label_info):
	# return semantic_map -> [H, W]
	semantic_map = np.zeros(label.shape[:-1])
	for index, info in enumerate(label_info):
		color = label_info[info]
		# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map[class_map] = index
		# semantic_map.append(class_map)
	# semantic_map = np.stack(semantic_map, axis=-1)
	return semantic_map


def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x


def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
	label_values.append([0, 0, 0])
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]

	return x

def compute_global_accuracy(pred, label):
	pred = pred.flatten()
	label = label.flatten()
	total = len(label)
	count = 0.0
	for i in range(total):
		if pred[i] == label[i]:
			count = count + 1.0
	return float(count) / float(total)

def fast_hist(a, b, n):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

class RandomCrop(object):
	"""Crop the given PIL Image at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

	def __init__(self, size, seed, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.seed = seed

	@staticmethod
	def get_params(img, output_size, seed):
		"""Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		random.seed(seed)
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.

		Returns:
			PIL Image: Cropped image.
		"""
		if self.padding > 0:
			img = torchvision.transforms.functional.pad(img, self.padding)

		# pad the width if needed
		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
		# pad the height if needed
		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.seed)

		return torchvision.transforms.functional.crop(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def cal_miou(miou_list, csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	miou_dict = {}
	cnt = 0
	for iter, row in ann.iterrows():
		label_name = row['name']
		class_11 = int(row['class_11'])
		if class_11 == 1:
			miou_dict[label_name] = miou_list[cnt]
			cnt += 1
	return miou_dict, np.mean(miou_list)

class OHEM_CrossEntroy_Loss(nn.Module):
	def __init__(self, threshold, keep_num):
		super(OHEM_CrossEntroy_Loss, self).__init__()
		self.threshold = threshold
		self.keep_num = keep_num
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def forward(self, output, target):
		loss = self.loss_function(output, target).view(-1)
		loss, loss_index = torch.sort(loss, descending=True)
		threshold_in_keep_num = loss[self.keep_num]
		if threshold_in_keep_num > self.threshold:
			loss = loss[loss>self.threshold]
		else:
			loss = loss[:self.keep_num]
		return torch.mean(loss)

def group_weight(weight_group, module, norm_layer, lr):
	group_decay = []
	group_no_decay = []
	for m in module.modules():
		if isinstance(m, nn.Linear):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
			if m.weight is not None:
				group_no_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)

	assert len(list(module.parameters())) == len(group_decay) + len(
		group_no_decay)
	weight_group.append(dict(params=group_decay, lr=lr))
	weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
	return weight_group

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def get_target_tensor(input_tensor, mode):
    # Source tensor = 0.0
    # Target tensor =  1.0
    source_tensor = torch.FloatTensor(1).fill_(0.0)
    target_tensor = torch.FloatTensor(1).fill_(1.0)
    source_tensor = source_tensor.expand_as(input_tensor)
    target_tensor = target_tensor.expand_as(input_tensor)
    if mode == 'source':
        return source_tensor
    elif mode == 'target':
        return target_tensor


def get_target_tensor_mc(input_tensor, mode):
    # Source tensor = 0.0
    # Target tensor =  1.0
    source_tensor = torch.FloatTensor(1).fill_(0.0)
    target_tensor = torch.FloatTensor(1).fill_(1.0)
    #source_tensor = source_tensor.expand_as(input_tensor)
    source_tensor = source_tensor.expand((input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3]))
    #target_tensor = target_tensor.expand_as(input_tensor)
    target_tensor = target_tensor.expand((input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3]))
    if mode == 'source':
        return source_tensor
    elif mode == 'target':
        return target_tensor
        
def save_da_model(args, model, model_d1, optimizer, optimizer_d1, epoch, name = None):
    if name is None:
      filename = args.checkpoint_name_save  
    else:
      filename = args.checkpoint_name_save.replace(".pth", "_best.pth")
    torch.save({
        'optimizer_state': optimizer.state_dict(),
        'optimizer_d_state': optimizer_d1.state_dict(),
        'model_state': model.state_dict(),
        'model_d_state': model_d1.state_dict(),
        'total_epoch_so_far': epoch
    }, os.path.join(args.save_model_path, filename))


def load_da_model(args, model, model_d1, optimizer, optimizer_d1):
    print("Resuming Model...")
    state_dict_path = os.path.join(args.save_model_path, args.checkpoint_name_load)
    checkpoint = torch.load(state_dict_path)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    optimizer_d1.load_state_dict(checkpoint['optimizer_d_state'])
    model.load_state_dict(checkpoint['model_state'])
    model_d1.load_state_dict(checkpoint['model_d_state'])
    epoch = checkpoint['total_epoch_so_far']
    return model, model_d1, optimizer, optimizer_d1, epoch


def save_model(args, model, optimizer, epoch, name = None):
    if name is None:
      filename = args.checkpoint_name_save  
    else:
      filename = args.checkpoint_name_save.replace(".pth", "_best.pth")
    torch.save({
        'optimizer_state': optimizer.state_dict(),
        'model_state': model.state_dict(),
        'total_epoch_so_far': epoch
    }, os.path.join(args.save_model_path, filename))


def load_model(args, model, optimizer):
    print("Resuming Model...")
    state_dict_path = os.path.join(args.save_model_path, args.checkpoint_name_load)
    checkpoint = torch.load(state_dict_path)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.load_state_dict(checkpoint['model_state'])
    epoch = checkpoint['total_epoch_so_far']
    return model, optimizer, epoch
