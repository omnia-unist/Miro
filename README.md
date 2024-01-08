# Miro
#### **Official Pytorch implementation of Cost-effective On-device Continual Learning over Memory Hierarchy with Miro**, accepted at **ACM MobiCom 2023**
[Paper](https://dl.acm.org/doi/10.1145/3570361.3613297) | [Slide](https://www.sigmobile.org/mobicom/2023/media/presentations/MaMiro.pdf)

This code is built atop [CarM (DAC '22)](https://dl.acm.org/doi/10.1145/3489517.3530587).


Abstract
-------------
Continual learning (CL) trains NN models incrementally from a continuous stream of tasks. To remember previously learned knowledge, prior studies store old samples over a memory hierarchy and replay them when new tasks arrive. Edge devices that adopt CL to preserve data privacy are typically energy-sensitive and thus require high model accuracy while not compromising energy efficiency, i.e., cost-effectiveness. Our work is the first to explore the design space of hierarchical memory replay-based CL to gain insights into achieving cost-effectiveness on edge devices. We present Miro, a novel system runtime that carefully integrates our insights into the CL framework by enabling it to dynamically configure the CL system based on resource states for the best cost-effectiveness. To reach this goal, Miro also performs online profiling on parameters with clear accuracy-energy trade-offs and adapts to optimal values with low overhead. Extensive evaluations show that Miro significantly outperforms baseline systems we build for comparison, consistently achieving higher cost-effectiveness.

Installation
-------------
Requirements for Anaconda is provided. We recommend to run Miro with conda env.   
```console
conda install --name miro --file environment.yml
```

Run Experiments
-------------
```console

# CIFAR-100 
python main.py --config=./config/cifar/er_cifar100_with_miro.yml

# Tiny-ImageNet
python main.py --config=./config/tiny/er_tiny_with_miro.yml

# Daily Sports and Activities
python main.py --config=./config/dsads/er_dsads_with_miro.yml

# UrbanSound8K (mel spectrogram)
python main.py --config=./config/us8k/er_us8k_with_miro.yml

# ImageNet-1k (50 Tasks)
python main.py --config=./config/imagenet1k/50tasks/er_img1k_with_miro.yml

```
