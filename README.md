# PGSR

> [Learning Piecewise Planar Representation for RGB Guided Depth Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10629189), in *IEEE Transactions on Computational Imaging* (TCI), 2024.
> Ruikang Xu, Mingde Yao, Yuanshen Guan, Zhiwei Xiong.

****
## Datesets

### Preparing
* The NYU_v2 dataset can be downloaded from this [link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
* The Middlebury dataset can be downloaded from this [link](https://vision.middlebury.edu/stereo/data/).
* The Lu dataset can be downloaded from this [link](https://web.cecs.pdx.edu/~fliu/project/depth-enhance/).
* The RGB-D-D dataset can be downloaded from this [link](http://mepro.bjtu.edu.cn/resource.html).

### Partitioning
* Training Set: We taking the first 1000 pairs from the [NYU_v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset as the training set and use the same preprocessing as [FDSR](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.pdf) and [DCTNet](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Discrete_Cosine_Transform_Network_for_Guided_Depth_Map_Super-Resolution_CVPR_2022_paper.html).
* Test Set: We use the rest 449 pairs from [NYU_v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [Middlebury](https://vision.middlebury.edu/stereo/data/), [Lu](https://web.cecs.pdx.edu/~fliu/project/depth-enhance/) and [RGB-D-D](http://mepro.bjtu.edu.cn/resource.html) as the testing set.

****
## Dependencies

* Python 3.8.8, PyTorch 1.8.0, torchvision 0.9.0.
* NumPy 1.24.2, OpenCV 4.7.0, Tensorboardx 2.5.1, kornia, Pillow, Imageio.
  
****
## Quick Start

### Inference
```
cd ./src && python test.py
```

### Training
```
cd ./src && python train.py
```

****
## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn.

****
## Citation
If you find our work helpful, please cite the following paper.
```
@article{xu2024learning,
  title={Learning Piecewise Planar Representation for RGB Guided Depth Super-Resolution},
  author={Xu, Ruikang and Yao, Mingde and Guan, Yuanshen and Xiong, Zhiwei},
  journal={IEEE Transactions on Computational Imaging},
  year={2024},
  publisher={IEEE}
}
```
