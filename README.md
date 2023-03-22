# SBLNP Algorithm
## Introduction

Accurate prediction of lymph node metastasis (LNM) status is important for the diagnosis and treatment of patients with muscle-invasive bladder cancer (MIBC). In this multicenter study, we developed a weakly-supervised deep learning-based model (named SBLNP) to predict LNM status from routine H&E-stained slides of primary muscle invasive bladder cancer (MIBC), and attempted to identify new histopathological features. Our results demonstrated that SBLNP performed well in three independent cohorts, showing excellent generalization ability. The combined classifier based on SBLNP and clinicopathologic variables demonstrated satisfactory performance. Interestingly, the SBLNP generated a new biological hypothesis, defining the lymphocytic inflammatory stroma as a key factor for prediction. To our knowledge, this is the first study that links inflammatory infiltrate in the stroma of MIBC to the LNM status. With the assistance of the SBLNP, clinicians are expected to identify appropriate population for neoadjuvant chemotherapy and pelvic lymph node dissection, avoiding the risks of under- or overtreatment.

https://github.com/zqy396/SBLNP/files/11042899/Figure.2.pdf

## Pre-requisites:

- Linux (Tested on Ubuntu 18.04)
 
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3060 Ti x 2 on local workstations)
 
- Python (3.8.12), h5py (3.6.0), matplotlib (3.5.2), numpy (1.23.5), opencv-python (4.5.5.64), openslide-python (1.1.2), pandas (1.4.2), pillow (9.0.1), PyTorch (1.10.0+cu111), scikit-learn (1.1.3), scipy (1.3.1),  tensorboardx (2.5), torchvision (0.11.0+cu111), smooth-topk, histolab (0.5.1).

## WSI Acquisition

You can start your GDC (https://portal.gdc.cancer.gov/projects/TCGA-BLCA) in the portal to download TCGA_BLCA clinical data and WSI repeat in this study.

## WSI Patching

Using the openslide-python toolkit to crop patches with a size of 448 × 448 pixels for each WSI.

`python histolab_a.py`

## Extract Features Using Resnet50

Extracte 2048 relevant features for each patch using a ResNet-50 neural network.

`python Extract2048.py`

## Reduce Dimensions

Using an adaptive encoder for dimensionality reduction, reducing the 2048 dimensions extracted from ResNet-50 to 512 dimensions.

`python AE.py`

## Train

SBLNP is an end-to-end weakly-supervised deep learning model, an advanced binary classification network based on multiple instance learning (MIL) and attention mechanism. See the article for more training details.

`python train.py`

## Test

`python eval.py`

The results will be stored in the output folder as set.
