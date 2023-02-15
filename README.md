# MV-DEFEAT

This repository provides the code for our paper "MV-DEFAT:Multi-View Deep Evidential Neural Network For Assessment Of Mammograms"

$\color[rgb]{1,0,0} Notice, this repository is only for the reviwer purpose, we will open-source the complete working code after the manuscript acceptance. $




### Introduction

We proposed a multi-view deep evidential neural network approach for assessment of mammograms by aggregating multiple views of a mammogram and utilizing the Dempster-Shafer evidential learning and combination rule to handle uncertain or conflicting evidence. 

The main objectives of the proposed approach are to enhance the precision and reliability of the mammogram density assessment by using a multi-view method and an
evidential optimization loss function. The main contributions of this study are as follows:
1. Performing extensive experiments to identify the optimal pre-trained CNN model for mammogram assessment task.
2. Extending the pre-trained CNN backbone architecture by incorporating evidential layers.
3. Developing a multi-view deep evidential neural network trained with a multi-view evidential loss function
4. Demonstrating the proposed model’s generalization and transferability capabilities on unseen datasets.

We conducted experiments on two open-source digital mammogram datasets, VinDr-Mammo and DDSM, which include 4,977 and 1,885 mammogram examinations, respectively. In addition, we validated MV-DEFEAT generalization and transferability capabilities on two independent datasets, CMMD and VTB, with 826 and 765 mammogram examinations, respectively.

### Requirements

python
Pytorch (enabled with CUDA), torchvision
Matplotlib, seaborn, scikitplot
scikit-learn
numpy
pandas
munch
yml
tqdm
Pillow
timm 

### Training and evaluation

Please, change the config file path for dataset path. Create log file folder to save the model path. 

Training: python train.py --config_path ddsm_ipsilateral_config.yml
Evaluation: python evaluate.py


### Acknowledgement

Part of the code is borrowed from https://github.com/hanmenghan/TMC



