# Multiple Layer Self Supervised Learning for Real Time Semantinc Segmentation

To run the classical training of the bisenet:
```
!pip install tensorboardX
!python train.py
```

To run the unsupervised version:
```
!python trainDAUnsupervised.py
```

## Model complexity

It is possible to print MACs, FLOPs and number of parameters of a model in this repository by using this script:
```
!pip install thop==0.0.5-2110061705
!python flops_calculator.py
```

## Model complexity

It is possible to print MACs, FLOPs and number of parameters of :
```
!pip install thop
!python complexity.py
```
