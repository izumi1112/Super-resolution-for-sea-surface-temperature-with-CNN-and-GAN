# Super-resolution-for-sea-surface-temperature-with-CNN-and-GAN
## Installation
- Python 3.8
- PyTorch == 1.7
- NVIDIA GPU + CUDA
- Python packages: `pip install numpy opencv-python lmdb pyyaml tb-nightly future`
## Datasets
### ERA20C
https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-20c
### OISST
https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
## Train and Test example
### Train
`python train.py -opt options/train/jp1/train_SRCNN_LR2HR.yml`
### Test
`python test.py -opt options/test/jp1/test_SRCNN_LR2HR.yml`
## Evaluation metric
### Learned Perceptual Image Patch Similarity
https://github.com/richzhang/PerceptualSimilarity
### Perceptual Index
https://github.com/roimehrez/PIRM2018
