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

- [ ] Data Augmentation
- [ ] Back-propagation
- [ ] Mini batch Stochastic Gradient Descent with Momentum
- [ ] Weight Initialization
- [ ] Learning Rate
- [ ] Early Stopping

## Experiments

### Standard Operating Condition (SOC)

You can download from [MSTAR Overview](https://www.sdms.afrl.af.mil/index.php?collection=mstar)

- MSTAR Target Chips (T72 BMP2 BTR70 SLICY) which is **MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY.zip**
- MSTAR / IU Mixed Targets which consists of **MSTAR-PublicMixedTargets-CD1.zip** and **
  MSTAR-PublicMixedTargets-CD2.zip**
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

### Extended Operating Conditions (EOC)

### Outlier Rejection

### End-to-End SAR-ATR Cases

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

## TODO

- [ ] Implementation
    - [ ] Data generation
        - [ ] SOC
        - [ ] EOC
        - [ ] Outlier Rejection
        - [ ] End-to-End SAR-ATR
    - [ ] Data Loader
        - [ ] SOC
        - [ ] EOC
        - [ ] Outlier Rejection
        - [ ] End-to-End SAR-ATR
    - [ ] Model
        - [ ] Network
        - [ ] Training
        - [ ] Early Stopping
        - [ ] Hyper-parameter Optimization
    - [ ] Experiments
        - [ ] Reproduce the SOC Results
            - [ ] 1 channel input (Magnitude only)
            - [ ] 2 channel input (Magnitude + Phase)
        - [ ] Reproduce the EOC Results
        - [ ] Reproduce the outlier rejection
        - [ ] Reproduce the end-to-end SAR-ATR

