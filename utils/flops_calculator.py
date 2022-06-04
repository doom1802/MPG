#from ..model.build_BiSeNet import BiSeNet

#from pytorch_model_summary import summary
#from fvcore.nn.flop_count import FlopCountAnalysis
import torch
from torch import randn
from thop import profile
from thop import clever_format


def complexity(model, input):
    macs, params = profile(model, inputs=(input, ), verbose=False)
    # MACs: multiply–accumulate operations, counts the number of a+(bxc) operations
    # FLOPs: floating point operations, counts the number of add/sub/div/mul operations
    # Since each MAC operation is composed by an add and a mult, FLOPs are nearly two times as MACs
    flops = 2*macs
    return macs, flops, params

def get_complexity(model, model_d, model_d_light):
  #model = BiSeNet(19, 'resnet18')
  #model_d = FCDiscriminator(19)
  #model_d_light = FCDiscriminatorLight(19)
  #print(summary(model, torch.zeros((1, 3, 512, 1024))))
  #print(summary(model_d, torch.zeros((1, 19, 512, 1024))))
  #print(summary(model_d_light, torch.zeros((1, 19, 512, 1024))))
  #flops = FlopCountAnalysis(model, torch.zeros((2, 3, 512, 1024)))
  #flops = FlopCountAnalysis(model_d, torch.zeros((2, 3, 512, 1024)))
  #flops = FlopCountAnalysis(model_d_light, torch.zeros((2, 3, 512, 1024)))
  input = randn(1, 3, 512, 1024)
  macs, flops, params = complexity(model, input)
  macs, flops, params = clever_format([macs, flops, params], "%.3f")
  
  print(f"\nCOMPLEXITY OF Bisenet")
  
  print(f"MACs: {macs}")
  print(f"FLOPs: {flops}")
  print(f"Parameters: {params}\n")

  input = randn(1, 19, 512, 1024)
  macs, flops, params = complexity(model_d, input)
  macs, flops, params = clever_format([macs, flops, params], "%.3f")
  print(f"COMPLEXITY OF FCDiscriminator")
  print(f"MACs: {macs}")
  print(f"FLOPs: {flops}")
  print(f"Parameters: {params}\n")
  
  input = randn(1, 19, 512, 1024)
  macs, flops, params = complexity(model_d_light, input)
  macs, flops, params = clever_format([macs, flops, params], "%.3f")
  
  print(f"COMPLEXITY OF FCDiscriminatorLight")
  print(f"MACs: {macs}")
  print(f"FLOPs: {flops}")
  print(f"Parameters: {params}")

if __name__ == '__main__':
    main()