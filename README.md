# Unsupervised domain adaptation with multi-level distillation boost and adaptive mask for medical image segmentation
**Pytorch implementation for Medical Image Analysis journal paper "Unsupervised domain adaptation with multi-level distillation boost and adaptive mask for medical image segmentation".**

## Requirements
Pytorch 2.0.1+cu118

## Code Stucture
The important files of this project is as following:
```
  - da_train.py: entry file for pre-training the checkpoint on source domain data.
  - da_solver.py: details for training the checkpoint in pre-training stage.
  - mt_da_train.py: entry file for adaptation stage, accepts all args.
  - mt_da_solver.py: details for training the checkpoint in adaptation stage.
    
  /data_loader: data loader files and for datasets.
```
## Implementation
### Train on datasets in the paper.
For training detail of each dataset, please check the ```scripts/```

Please store the each dataset in the following path:

**Refuge**
```
/data
  /refuge
    /train
      /images
      /masks
```

**RIM**
```
/data
  /rim
    /images
    /masks
```
**RIGAPlus**
```
/MLDB_AMIC
/RIGAPlus
```

**Drishti-GS**
```
/Drishti-GS
  /train
    /images
    /masks
  /test
    /images
    /masks
```
**CVC-Endo**
```
/polyp/
  /CVC-Endo
    /Train
      /images
      /masks
    /Test
      /images
      /masks
    /Valid
      /images
      /masks
```
**ETIS-LaribPolypDB**
```
/polyp/
  /ETIS-LaribPolypDB
    /images
    /masks
```
**CVC-ColonDB**
```
/polyp/
  /CVC-ColonDB
    /images
    /masks
```
**kvasir**
```
/polyp/
  /kvasir
    /images
    /masks
```


### Run pre-training stage:
```
bash scripts/run.sh 1 4
```
### Run adaptation stage:
```
bash scripts/run.sh 100 4 "pre-trained_checkpoint.pth"
```
View details in scripts/adapt.sh

### Test:
```
add "--test_only" as arg in /scripts/run.sh for one of test case.
```
### Inference:
```
add both "--test_only" and "--inference_only" as args in /scripts/run.sh for one of test case.
```

## Data
In order to train and test the model, you first need to download the [Drishti-GS](https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation),
 [RIM-ONE](http://medimrg.webs.ull.es/research/downloads/),
[refuge](https://refuge.grand-challenge.org/) datasets,
or
[CVC-EndoSceneStill](https://drive.google.com/file/d/1MuO2SbGgOL_jdBu3ffSf92feBtj8pbnw/view?usp=sharing),
[Kvasir-SEG](https://drive.google.com/file/d/1S9aV_CkvJcsouRN4zvjtyL1vDhBkGRqA/view?usp=sharing),
[ETIS-LaribPolypDB](https://www.dropbox.com/s/j4nsxijf5dhzb6w/ETIS-LaribPolypDB.rar?dl=0&file_subpath=%2FETIS-LaribPolypDB)
and 
[others](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579?file=37636550)
and place them in the folder path ' ./data '.
Place [RIGA+](https://zenodo.org/records/6325549) in a directory at the same level as the project folder.

## Pretrained model
- The predictions can be downloaded at [here](https://drive.google.com/file/d/1c_LTUu0rjv9XCMewoGsqSvmhXGLMMCQE/view?usp=drive_link)
- The dataset Refuge pre-trained checkpoints can be downloaded at:
   - [Auxiliary branch 4](https://drive.google.com/file/d/1Kpd81rEjbbSLsbNDo6eI1J-nf7BIvlC4/view?usp=drive_link)
   - [Auxiliary branch 3,4](https://drive.google.com/file/d/1_b70A-4OdoDZM1-nsGhFbxz_yBV57OY1/view?usp=drive_link)
- The dataset BinRushed & Magrabia pre-trained checkpoints can be downloaded at:
   - [Auxiliary branch 4](https://drive.google.com/file/d/1QPfE9dG4BBWVV7H2_6piN-z9hYSNi1AK/view?usp=drive_link)
