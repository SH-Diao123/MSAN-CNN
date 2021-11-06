# MSAN-CNN



This is a PyTorch implementation of the paper 'Weakly-supervised framework for cancer regions detection of hepatocellular carcinoma in whole-slide pathological images based on multi-scale attention convolutional neural network',  and We'll refine the data and code over time.


![overview](https://github.com/SH-Diao123/MSAN-CNN/blob/main/assets/overview.png)

## 🤝 Authorization 
If you would like to use our data, please contact us first and obtain authorization to use it.


## 🤝 Citation

If you find this code is useful for your research, please consider citing:

```javascript 
@article{
title={Weakly-supervised framework for cancer regions detection of hepatocellular carcinoma in whole-slide pathological images based on multi-scale attention convolutional neural network}，
author={Songhui Diao, Yinli Tian, Wanming Hu, Jiaxin Hou, Ricardo Lambo, Zhicheng Zhang, Yaoqin Xie, Xiu Nie, Fa Zhang, Racoceanu Daniel, Wenjian Qin}，
journal={The American Journal of Pathology}，
year={2021},
}
```

## Setup
### Prerequisites
- PyTorch 1.9.0
- python 3.8.5
- Torchvision 0.10.0
- numpy and so on
### Data
- train data
- validation data
- test data

## Training and Validation
Training a network with default arguments. Model checkpoints and tensorboard logs are written out to a unique directory created by default within experiments/models and experiments/logs respectively after starting training.
If conditions permit, it will be better to pre-train the model first.
```javascript 
python main.py
```

## Testing
You can run validation and testing on the checkpointed best model by:
```javascript 
python main.py
```









