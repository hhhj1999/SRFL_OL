## Requirements
#### numpy                   1.21.5
#### Pillow                  9.2.0
#### scikit-image            0.16.2
#### scipy                   1.7.3
#### tensorboard             2.10.0
#### tensorboardX            2.0
#### timm                    0.5.4
#### torch                   1.7.0+cu101
#### tqdm                    4.41.1


## Datasets
#### Download the [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) 
#### Download the [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) 
#### Download the [NABirds](https://dl.allaboutbirds.org/nabirds) 
#### You can also try other fine-grained datasets. 

## Training
Run ``python train.py``. You may need to change the configurations in ``config.py`` or ``tiny_vit.py`` in networks.

## Evaluation
Predictions can be made after each training step, or via test.py

## Reference
This project is based on the following implementation:

[TransFG](https://github.com/TACJu/TransFG)
[MMAL](https://github.com/ZF4444/MMAL-Net)
[SIM](https://github.com/pku-icst-mipl/sim-trans_acmmm2022)
[TinyVit](https://github.com/microsoft/Cream/tree/main/TinyViT)

## Note
The README.md will be refined later