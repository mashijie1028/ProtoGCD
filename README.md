# ProtoGCD: Unified and Unbiased Prototype Learning for Generalized Category Discovery

<a href='https://ieeexplore.ieee.org/document/10948388'><img src='https://img.shields.io/badge/-TPAMI%202025-purple'></a> <a href='https://arxiv.org/abs/2504.03755'><img src='https://img.shields.io/badge/ArXiv-2504.03755-red'></a> 

Official implementation of our TPAMI 2025 paper "ProtoGCD: Unified and Unbiased Prototype Learning for Generalized Category Discovery".

![method](assets/method.jpg)

## :running: ​Running

### Dependencies

```
loguru
numpy
pandas
scikit_learn
scipy
torch==1.10.0
torchvision==0.11.1
tqdm
```

### Datasets

We conduct experiments on 7 datasets:

* Generic datasets: CIFAR-10, CIFAR-100, ImageNet-100
* Fine-grained datasets: [CUB](https://drive.google.com/drive/folders/1kFzIqZL_pEBVR7Ca_8IKibfWoeZc3GT1), [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

### Training





## :clipboard: ​Citing this work

```bibtex
@ARTICLE{10948388,
  author={Ma, Shijie and Zhu, Fei and Zhang, Xu-Yao and Liu, Cheng-Lin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ProtoGCD: Unified and Unbiased Prototype Learning for Generalized Category Discovery}, 
  year={2025},
  volume={},
  number={},
  pages={1-17},
  keywords={Prototypes;Adaptation models;Contrastive learning;Training;Magnetic heads;Feature extraction;Estimation;Automobiles;Accuracy;Pragmatics;Generalized category discovery;open-world learning;prototype learning;semi-supervised learning},
  doi={10.1109/TPAMI.2025.3557502}
}
```



## :gift: ​Acknowledgements

In building the AGCD codebase, we reference [SimGCD](https://github.com/CVMI-Lab/SimGCD).



## :white_check_mark: ​License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mashijie1028/ProtoGCD/blob/main/LICENSE) file for details.



## :email: ​Contact

If you have further questions or discussions, feel free to contact me:

Shijie Ma (mashijie2021@ia.ac.cn)