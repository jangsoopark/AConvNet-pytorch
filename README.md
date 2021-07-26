# AConvNet

### Target Classification Using the Deep Convolutional Networks for SAR Images

This repository is reproduced-implementation of AConvNet which recognize target from MSTAR dataset.
You can see the official implementation of the author at [MSTAR-AConvNet](https://github.com/fudanxu/MSTAR-AConvNet).

## Dataset

### MSTAR (Moving and Stationary Target Acquisition and Recognition) Database

#### Format

- Header
    - Type: ASCII
    - Including data shape(width, height), serial number, azimuth angle, etc.
- Data
    - Type: Two-bytes
    - Shape: W x H x 2
        - Magnitude block
        - Phase Block

Below figure is the example of magnitude block(Left) and phase block(Right)

![Example of data block: 2S1](./assets/figure/001.png)

## Model

The proposed model only consists of **sparsely connected layers** without any fully connected layers.

- It eases the over-fitting problem by reducing the number of free parameters(model capacity)

|    layer    | Input  |   Conv 1   |   Conv 2   |   Conv 3   | Conv 4 | Conv 5  |
| :---------: | ------ | :--------: | :--------: | :--------: | :----: | :-----: |
|  channels   | 2      |     16     |     32     |     64     |  128   |   10    |
| weight size | -      |   5 x 5    |   5 x 5    |   6 x 6    | 5 x 5  |  3 x 3  |
|   pooling   | -      | 2 x 2 - s2 | 2 x 2 - s2 | 2 x 2 - s2 |   -    |    -    |
|   dropout   | -      |     -      |     -      |     -      |  0.5   |    -    |
| activation  | linear |    ReLU    |    ReLU    |    ReLU    |  ReLU  | Softmax |

## Training
For training, this implementation fixes the random seed to `12321` for `reproducibility`.

The experimental conditions are same as in the paper, except for `data augmentation` and `learning rate`. 
The `learning rate` is initialized with `1e-3` and decreased by a factor of 0.1 **after 26 epochs**.
You can see the details in `src/model/_base.py` and `experiments/config/AConvNet-SOC.json`

### Data Augmentation
 
- The author uses random shifting to extract 88 x 88 patches from 128 x 128 SAR image chips.
    - The number of training images per one SAR image chip could be increased at maximum by (128 - 88 + 1) x (128 - 88 + 1) = 1681.

- However, for SOC, this repository does not use random shifting tue to accuracy issue.
    - You can see the details in `src/data/generate_dataset.py` and `src/data/mstar.py`
    - This implementation failed to achieve higher than 98% accuracy when using random sampling.
    - The implementation details for data augmentation is as: 
        - Crop the center of 94 x 94 size image on 128 x 128 SAR image chip (49 patches per image chip).
        - Extract 88 x 88 patches with stride 1 from 94 x 94 image.
    

## Experiments

You can download the MSTAR Dataset from [MSTAR Overview](https://www.sdms.afrl.af.mil/index.php?collection=mstar)

### Standard Operating Condition (SOC)

- MSTAR Target Chips (T72 BMP2 BTR70 SLICY) which is **MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY.zip**
- MSTAR / IU Mixed Targets which consists of **MSTAR-PublicMixedTargets-CD1.zip** and **MSTAR-PublicMixedTargets-CD2.zip**
- **SLICY target is ignored**

|         |            | Train      |            | Test       |            |
| ------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Class   | Serial No. | Depression | No. Images | Depression | No. Images |
| BMP-2   | 9563       | 17         | 233        | 15         | 196        |
| BTR-70  | c71        | 17         | 233        | 15         | 196        |
| T-72    | 132        | 17         | 232        | 15         | 196        |
| BTR-60  | k10yt7532  | 17         | 256        | 15         | 195        |
| 2S1     | b01        | 17         | 299        | 15         | 274        |
| BRDM-2  | E-71       | 17         | 298        | 15         | 274        |
| D7      | 92v13015   | 17         | 299        | 15         | 274        |
| T-62    | A51        | 17         | 299        | 15         | 273        |
| ZIL-131 | E12        | 17         | 299        | 15         | 274        |
| ZSU-234 | d08        | 17         | 299        | 15         | 274        |

#### Training Set (Depression: 17$\degree$​)

```shell
MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY
├ TRAIN/17_DEG
│    ├ BMP2/SN_9563/*.000 (233 images)
│    ├ BTR70/SN_C71/*.004 (233 images)
│    └ T72/SN_132/*.015   (232 images)
└ ...

MSTAR-PublicMixedTargets-CD2/MSTAR_PUBLIC_MIXED_TARGETS_CD2
├ 17_DEG
│    ├ COL1/SCENE1/BTR_60/*.003  (256 images)
│    └ COL2/SCENE1
│        ├ 2S1/*.000            (299 images)
│        ├ BRDM_2/*.001         (298 images)
│        ├ D7/*.005             (299 images)
│        ├ SLICY
│        ├ T62/*.016            (299 images)
│        ├ ZIL131/*.025         (299 images)
│        └ ZSU_23_4/*.026       (299 images)
└ ...

```

#### Test Set (Depression: 15$\degree$​​)

```shell
MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY
├ TEST/15_DEG
│    ├ BMP2/SN_9563/*.000 (195 images)
│    ├ BTR70/SN_C71/*.004 (196 images)
│    └ T72/SN_132/*.015   (196 images)
└ ...

MSTAR-PublicMixedTargets-CD1/MSTAR_PUBLIC_MIXED_TARGETS_CD1
├ 15_DEG
│    ├ COL1/SCENE1/BTR_60/*.003  (195 images)
│    └ COL2/SCENE1
│        ├ 2S1/*.000            (274 images)
│        ├ BRDM_2/*.001         (274 images)
│        ├ D7/*.005             (274 images)
│        ├ SLICY
│        ├ T62/*.016            (273 images)
│        ├ ZIL131/*.025         (274 images)
│        └ ZSU_23_4/*.026       (274 images)
└ ...

```
#### Quick Start Guide for Training

- Dataset Preparation
    - Download the [soc-dataset.zip](https://github.com/jangsoopark/AConvNet-pytorch/releases/download/V2.0.0/soc-raw.zip) 
    - After extracting it, you can find `train` and  `test` directories inside `raw` directory.
    - Place the two directories (`train` and  `test`) to the `dataset/raw`.
```shell
$ cd src/data 
$ python3 generate_dataset.py --is_train=True --use_phase=True
$ python3 generate_dataset.py --is_train=False --use_phase=True
$ cd ..
$ python3 train.py
```

#### Results of SOC
- Final Accuracy is **99.18%** (The official accuracy is 99.13%)
- You can see the details in `notebook/experiments-SOC.ipynb`

- Visualization of training loss and test accuracy

![soc-training-plot](./assets/figure/soc-training-plot.png)

- Confusion Matrix with best model at **epoch 28**

![soc-confusion-matrix](./assets/figure/soc-confusion-matrix.png)

- Noise Simulation [1]
    - i.i.d samples from a uniform distribution
    - This simulation does not fix the random seed

| Noise | 1% | 5% | 10% | 15%|
| :---: | :---: | :---: | :---: | :---: |
| AConvNet-PyTorch | 98.56 | 94.39 | 85.03 | 73.65 |
| AConvNet-Official | 91.76 | 88.52 | 75.84 | 54.68 |


### Extended Operating Conditions (EOC)

### Outlier Rejection

### End-to-End SAR-ATR Cases

## Details about the specific environment of this repository

| | |
| :---------: | :------: |
| OS | Ubuntu 20.04 LTS |
| CPU | Intel i7-10700k |
| GPU | RTX 2080Ti 11GB |
| Memory | 32GB |
| SSD | 500GB |
| HDD | 2TB |

## Citation

```bibtex
@ARTICLE{7460942,
  author={S. {Chen} and H. {Wang} and F. {Xu} and Y. {Jin}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Target Classification Using the Deep Convolutional Networks for SAR Images}, 
  year={2016},
  volume={54},
  number={8},
  pages={4806-4817},
  doi={10.1109/TGRS.2016.2551720}
}
```

## References
[1] G. Dong, N. Wang, and G. Kuang, 
"Sparse representation of monogenic signal: With application to target recognition in SAR images,"
*IEEE Signal Process. Lett.*, vol. 21, no. 8, pp. 952-956, Aug. 2014.


---

## TODO

- [ ] Implementation
    - [ ] Data generation
        - [X] SOC
        - [ ] EOC
        - [ ] Outlier Rejection
        - [ ] End-to-End SAR-ATR
    - [ ] Data Loader
        - [X] SOC
        - [ ] EOC
        - [ ] Outlier Rejection
        - [ ] End-to-End SAR-ATR
    - [ ] Model
        - [X] Network
        - [X] Training
        - [X] Early Stopping
        - [X] Hyper-parameter Optimization
    - [ ] Experiments
        - [X] Reproduce the SOC Results
        - [ ] Reproduce the EOC Results
        - [ ] Reproduce the outlier rejection
        - [ ] Reproduce the end-to-end SAR-ATR

