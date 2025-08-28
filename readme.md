# DG-DDM-Seg
This repository is the official PyTorch implementation of the paper "Domain-Generalized Discrete Diffusion Model for Cross-Domain Medical Image Segmentation" in IEEE TMI 2025.




## Overview

<div align=center>
<img src="https://github.com/HeranYang/DG-DDM-Seg/blob/main/image/framework.png" width="800px"/>
</div>

Domain shift is a significant challenge in medical image segmentation, primarily due to variations in image acquisition protocols, modalities, etc. Domain shift often causes models trained on a source domain to perform poorly on unseen target domains. In this work, we introduce the Domain-Generalized Discrete Diffusion Model for Segmentation (DG-DDM-Seg), a diffusion-based generative model designed for single-source domain generalization in medical image segmentation. DG-DDM-Seg generates discrete conditional distributions of segmentation masks. To ensure domain independence, we employ two key strategies: 1) We extract robust features from conditional images to enhance the domain independence of diffusion model. 2) We use both conditional images and pseudo-labels as inputs to improve cross-domain segmentation performance. Along this idea, we propose a two-path reverse diffusion process during training, utilizing Robust Feature Extraction Subnet and Mask-Generation Transformer to learn a domain-generalized discrete conditional distribution based on robust image features and pseudo-labels. This learned distribution is then used to generate segmentation masks for unseen target domains. Experimental results demonstrate that DG-DDM-Seg achieves state-of-the-art performance in cross-domain medical image segmentation, with domain shifts in modality, sequence, and site.

The outline of this readme file is:

    Overview
    Requirements
    Dataset
    Usage
    Citation
    Reference
    
The folder structure of our implementation is:

    abdominal_ct2mr\       : code of DG-DDM-Seg for cross-modality abdominal segmentation (train on CT and test on MR)
    cardiac\               : code of DG-DDM-Seg for cross-sequence cardiac segmentation (train on bSSFP MR and test on LGE MR)
    prostate_a2rest\       : code of DG-DDM-Seg for cross-site prostate segmentation (train on site A and test on rest five sites)




## Requirements
All experiments utilize the PyTorch library. We recommend the following package versions:
* python==3.9
* pytorch==2.2.2 
* torchvision==0.17.2 
* torchaudio==2.2.2 
* pytorch-cuda=11.8
* opencv-python==4.10.0.84
* tensorboardX
* scipy
* numpy==1.24.0




## Dataset
... to be continued ...


### Data Preprocessing
... to be continued ...


### Data Folder Structure
The structure of our data folder is:





## Usage

We provide the codes of our DG-DDM-Seg respectively for cross-modality abdominal segmentation, cross-sequence cardiac segmentation, and cross-site prostate segmentation.
Note that the codes for these three tasks are identical except for the hyper-parameters.

The structure of our code folder is: 
(here we take abdominal_ct2mr as example)

    abdominal_ct2mr\            : code of DG-DDM-Seg for cross-modality abdominal segmentation
            |-- config\
            |        |-- our_train.json            : code of training setting
            |        |-- our_valid_mp.json         : code of validation setting
            |        |-- our_test_mp.json          : code of test setting
            |-- core\
            |        |-- logger.py
            |        |-- metrics.py
            |        |-- wandb_logger.py
            |-- data\
            |        |-- __init__.py
            |        |-- image_transforms.py
            |        |-- location_scale_augmentation.py
            |        |-- MRI_dataset.py
            |        |-- prepare_data.py
            |        |-- saliency_balancing_fusion.py
            |        |-- transform_utils.py
            |        |-- util.py
            |-- model\
            |        |-- basic_modules\
            |        |        |-- clip_grad_norm.py
            |        |        |-- common.py
            |        |        |-- misc.py
            |        |-- vqdm_modules\
            |        |        |-- diffusion_transformer.py
            |        |        |-- embedding.py
            |        |        |-- transformer_utils.py
            |        |-- __init__.py
            |        |-- model.py
            |        |-- networks.py
            |-- our_train.py            : code of training model
            |-- our_valid_mp.py         : code of model validation
            |-- our_test_mp.py          : code of testing model




### Training

Our code can be trained using the following commond:

    python our_train.py -p train -c config/our_train.json

If you want to continue train the model, you could modify the "resume_state" value in our_train.json to be the newest checkpoint path, and then run the commond above.


### Validation

The validation process can be conducted using the following commond:

    python our_valid_mp.py -p val -c config/our_valid_mp.json
    
After generating the validation results, you could select the optimal checkpoint based on the performance on validation set.


### Test

Before starting the test process, you need to modify the "resume_state" value in our_test_mp.json to be the optimal checkpoint path.
Then, you can generate the test results using the following commond:

    python our_test_mp.py -p val -c config/our_test_mp.json


### About Trained Model
We have also uploaded our trained DG-DDM-Seg models in the three cross-domain segmentation tasks mentioned above, and one can directly use them for cross-domain image segmentation. Due to the restriction of github, the trained models are uploaded to the [Google Drive](???).



## Citation
If you find this code useful for your research, please cite our paper:
> @article{yang2025domain, 
> <br> title={Domain-Generalized Discrete Diffusion Model for Cross-Domain Medical Image Segmentation}, 
> <br> author={Yang, Heran and Hua, Wenbo and Xu, Zongben and Sun, Jian},
> <br> journal={IEEE Transactions on Medical Imaging},
> <br> doi={10.1109/TMI.2025.3564474},
> <br> year={2025}}



## Reference

... to be continued ...
