![alt text](resources/logo_small_80.png)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pytorch-3dunet/badges/license.svg)](https://anaconda.org/conda-forge/pytorch-3dunet)

# Custom 3D Unet for Human Vasculature 

This repository is a fork of the [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) project, extended and modified for the identification and mapping of Human Vasculature from HiP-CT image slices of human kidneys. It includes a custom data loader that reads data directly from folders instead of HDF5 files, and a custom evaluation metric based on the surface dice metric found here:

https://github.com/google-deepmind/surface-distance

## Custom DataLoader - FolderDataset

`FolderDataset` is a new data loader that is capable of loading images directly from folders. This is particularly useful when dealing with a large number of `.tif` image files, making it unnecessary to convert them into the HDF5 format. This custom data loader is seamlessly integrated into the existing 3D-UNet framework and can be used for both training and validation purposes.

## Custom Evaluation Metric - Surface Dice

The model evaluation is performed using the Surface Dice metric, a robust metric for assessing the quality of segmentation, especially in medical imaging scenarios. This metric evaluates the model based on the surface similarity between the predicted and ground truth segmentations. It is the primary evaluation metric used to assess competition submissions.

## Installation

Installation remains similar to the original repository with minor adjustments to accommodate the new custom components.

- Install the fork:
   ```bash
   pip install git+https://github.com/HamishMatthews/pytorch-3dunet.git
   ```


- Install surface-distance package:
   ```bash
   pip install git+https://github.com/google-deepmind/surface-distance.git
   ```
   
## Usage

The usage of the `train3dunet` and `predict3dunet` commands remains the same. However, with the custom `FolderDataset` dataloader, the images and labels can now be in a directory, with its path and subfolder names included in the '.yml' configuration. 

For detailed usage, refer to the sample configuration files for training and prediction, which now support the FolderDataset structure.

## Cite

If you use this code or the original `pytorch-3dunet` code for your research, please cite the original paper and repository.

```
@article {10.7554/eLife.57613,
article_type = {journal},
title = {Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
author = {Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, Sören and Wilson-Sánchez, David and Lymbouridou, Rena and Steigleder, Susanne S and Pape, Constantin and Bailoni, Alberto and Duran-Nebreda, Salva and Bassel, George W and Lohmann, Jan U and Tsiantis, Miltos and Hamprecht, Fred A and Schneitz, Kay and Maizel, Alexis and Kreshuk, Anna},
editor = {Hardtke, Christian S and Bergmann, Dominique C and Bergmann, Dominique C and Graeff, Moritz},
volume = 9,
year = 2020,
month = {jul},
pub_date = {2020-07-29},
pages = {e57613},
citation = {eLife 2020;9:e57613},
doi = {10.7554/eLife.57613},
url = {https://doi.org/10.7554/eLife.57613},
keywords = {instance segmentation, cell segmentation, deep learning, image analysis},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```