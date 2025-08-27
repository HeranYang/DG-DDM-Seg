# DG-DDM-Seg
This repository is the official PyTorch implementation of the paper "Domain-Generalized Discrete Diffusion Model for Cross-Domain Medical Image Segmentation" in IEEE TMI 2025.


## Overview


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



### Data Preprocessing



### Data Folder Structure




## Usage

We provide the codes of our DG-DDM-Seg respectively for cross-modality abdominal segmentation, cross-sequence cardiac segmentation, and cross-site prostate segmentation.
The structure of our code folder is:

    abdominal_ct2mr\           : code of Hyper-GAE for multi-modal MR image synthesis
          config\
                |-- our_train.json
                |-- our_valid_mp.json
                |-- our_test_mp.json
          core\
                |-- logger.py
                |-- metrics.py
                |-- wandb_logger.py
          data\
                |-- __init__.py
                |-- image_transforms.py
                |-- location_scale_augmentation.py
                |-- MRI_dataset.py
                |-- prepare_data.py
                |-- saliency_balancing_fusion.py
                |-- transform_utils.py
                |-- util.py
          model\
                basic_modules\
                      |-- clip_grad_norm.py
                      |-- common.py
                      |-- misc.py
                vqdm_modules\
                      |-- diffusion_transformer.py
                      |-- embedding.py
                      |-- transformer_utils.py
                |-- __init__.py
                |-- model.py
                |-- networks.py
          |-- our_train.py
          |-- our_valid_mp.py         : main function
          |-- our_test_mp.py        : code of building model, and train/valid/test




### Training

Our code can be trained using the following commond:

    python our_train.py -p train -c config/our_train.json


### Validation

Before starting the validation process, you may need to modify the information about valid set and epoch in valid function within model.py.
Then, the validation process can be conducted using the following commond:

    python our_valid_mp.py -p val -c config/our_valid_mp.json
    
After generating the validation results, you could select the optimal epoch_id based on the performance on validation set.


### Test

Before starting the test process, you need to set the epoch as the selected optimal epoch_id in test function within model.py.
Then, you can generate the test results using the following commond:

    python our_test_mp.py -p val -c config/our_test_mp.json


### About Trained Model
We have also uploaded our trained DG-DDM-Seg models, and one can directly use them for cross-domain image segmentation tasks. Due to the restriction of github, the trained models are uploaded to the [Google Drive](???).



## Citation
If you find this code useful for your research, please cite our paper:
> @article{yang2025domain, 
> <br> title={Domain-Generalized Discrete Diffusion Model for Cross-Domain Medical Image Segmentation}, 
> <br> author={Yang, Heran and Hua, Wenbo and Xu, Zongben and Sun, Jian},
> <br> journal={IEEE Transactions on Medical Imaging},
> <br> doi={10.1109/TMI.2025.3564474},
> <br> year={2025}}



## Reference

