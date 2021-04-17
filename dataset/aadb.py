"""
Photo Aesthetics Ranking Network with Attributes and Content Adaptation. ECCV 2016
A aesthetics and attributes database (AADB) which contains:

===> 0) 10,000 images in total, 8,500 (training), 500 (validation), and 1,000 (testing)
with
===> 1) aesthetic scores
and
===> 2) meaningful attributes assigned to each image
by multiple human rater.

paper reference: https://www.ics.uci.edu/~fowlkes/papers/kslmf-eccv16.pdf
code reference: https://github.com/aimerykong/deepImageAestheticsAnalysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.datasets import VisionDataset


class AADBDataset(VisionDataset):
    ...
