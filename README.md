<br />
<p align="center">
    <h1 align="center">
        PACE: marrying generalization in PArameter-efficient fine-tuning with Consistency rEgularization (NeurIPS 2024 Spotlight)
    </h1>

  <p align="center">
    <p align="center">
    <a href="https://scholar.google.com/citations?user=oGD-WMQAAAAJ"><strong>Yao Ni</strong></a>
    ,
    <a href="https://scholar.google.com/citations?user=cnVvh7AAAAAJ"><strong>Shan Zhang</strong></a>
    ,
    <a href="https://www.koniusz.com/"><strong>Piotr Koniusz</strong></a>
    </p>
  </p>
  
  <p align="center">
    <a href='https://arxiv.org/abs/2409.17137'>
      <img src='https://img.shields.io/badge/Paper-arXiv-80261B?style=flat&logo=Googledocs&logoColor=white' alt='Paper arXiv'>
    </a>
    <a href='https://maxwellyaoni.github.io/home/documents/PACE_Slides.pdf'>
      <img src='https://img.shields.io/badge/Slides-2AA26C?style=flat&logo=Slides&logoColor=white' alt='Slides'>
    </a>
    <a href='https://maxwellyaoni.github.io/home/documents/PACE_Poster.pdf'>
      <img src='https://img.shields.io/badge/Poster-2AA26C?style=flat&logo=Packt&logoColor=white' alt='Slides'>
    </a>
    <a href='https://www.youtube.com/watch?v=CkThbYQ9SxY'>
      <img src='https://img.shields.io/badge/Video-Youtube-FA1D1D?style=flat&logo=youtube&logoColor=white' alt='Video Youtube'>
    </a>
  </p>
  <p align="center">
    <img src="https://maxwellyaoni.github.io/home/documents/pace_pipeline.jpg" alt="Overview" width="100%">
    </p>
</p>
<br/>

### Below are the general knowledges discovered from our work:
#### 💡 Lower gradient norms improve model generalization.
#### 💡 Consistency regularization across different perturbations reduces gradient norms, improving generalization.
#### 💡 Consistency regularization on adapter features aligns fine-tuned models with pre-trained ones, preserving knowledge.

 

## Code for [PACE-Vision](https://github.com/MaxwellYaoNi/PACE/tree/main/Vision) is Released.

## Code for PACE-NLP is being reorganized.

## We will add PACE into huggingface soon.

## Citation

If you find the theories or code help your work, please kindly cite our paper:
```
@inproceedings{
ni2024pace,
title={{PACE}: marrying the generalization of {PA}rameter-efficient fine-tuning with Consistency rEgularization},
author={Yao Ni and Shan Zhang and Piotr Koniusz},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=cOuLbPhOT1}
}
```
