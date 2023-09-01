# A neural network for monocular semantic segmentation and scene flow estimation in an aquatic environment

<img src=demo/demo.gif> 

> 3D visualization of estimated depth and scene flow from two temporally consecutive images.  
> Intermediate frames are interpolated using the estimated scene flow. (fine-tuned model, tested on KITTI Benchmark)

This repository uses the official PyTorch implementation of the paper **Self-Supervised Monocular Scene Flow Estimation** as the starting point:  

&nbsp;&nbsp;&nbsp;[**Self-Supervised Monocular Scene Flow Estimation**](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hur_Self-Supervised_Monocular_Scene_Flow_Estimation_CVPR_2020_paper.pdf)  
&nbsp;&nbsp;&nbsp;[Junhwa Hur](https://hurjunhwa.github.io) and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp)  
&nbsp;&nbsp;&nbsp;*CVPR*, 2020 (**Oral Presentation**)  
&nbsp;&nbsp;&nbsp;[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hur_Self-Supervised_Monocular_Scene_Flow_Estimation_CVPR_2020_paper.pdf) / [Supplemental](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Hur_Self-Supervised_Monocular_Scene_CVPR_2020_supplemental.pdf) / [Arxiv](https://arxiv.org/abs/2004.04143)

- Contact: junhwa.hur[at]gmail.com  

The repository is adapted for use with the MODD2 and MODS datasets.
A segmentation head is added.

## Getting started
This code has been developed with Anaconda (Python 3.7), **PyTorch 1.2.0** and CUDA 10.0 on Ubuntu 16.04.  
Based on a fresh [Anaconda](https://www.anaconda.com/download/) distribution and [PyTorch](https://pytorch.org/) installation, following packages need to be installed:  

  ```Shell
  conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
  pip install tensorboard
  pip install pypng==0.0.18
  pip install colorama
  pip install scikit-image
  pip install pytz
  pip install tqdm==4.30.0
  pip install future
  ```

Then, please excute the following to install the Correlation and Forward Warping layer:
  ```Shell
  ./install_modules.sh
  ```

**For PyTorch version > 1.3**  
Please put the **`align_corners=True`** flag in the `grid_sample` function in the following files:
  ```
  augmentations.py
  losses.py
  models/modules_sceneflow.py
  utils/sceneflow_util.py
  ```

## Acknowledgement

Please cite the original paper if you use our source code.  

```bibtex
@inproceedings{Hur:2020:SSM,  
  Author = {Junhwa Hur and Stefan Roth},  
  Booktitle = {CVPR},  
  Title = {Self-Supervised Monocular Scene Flow Estimation},  
  Year = {2020}  
}
```

- Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [Jochen Gast](https://scholar.google.com/citations?user=tmRcFacAAAAJ&hl=en)  
- MonoDepth evaluation utils from [MonoDepth](https://github.com/mrharicot/monodepth)
- MonoDepth PyTorch Implementation from [OniroAI / MonoDepth-PyTorch](https://github.com/OniroAI/MonoDepth-PyTorch)

