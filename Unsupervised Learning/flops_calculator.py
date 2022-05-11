from model.build_BiSeNet import BiSeNet
from pytorch_model_summary import summary
from fvcore.nn.flop_count import FlopCountAnalysis
import torch
from model.discriminator import FCDiscriminator
from model.discriminator_light import FCDiscriminatorLight

model = BiSeNet(19, 'resnet18')
model_d = FCDiscriminator(19)
model_d_light = FCDiscriminatorLight(19)
print(summary(model, torch.zeros((1, 3, 512, 1024))))
print(summary(model_d, torch.zeros((1, 19, 512, 1024))))
print(summary(model_d_light, torch.zeros((1, 19, 512, 1024))))
flops = FlopCountAnalysis(model, torch.zeros((2, 3, 512, 1024)))
print(flops.total())