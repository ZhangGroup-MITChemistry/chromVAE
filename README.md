# chromVAE: variational autoencoder for chromatin images

ChromVAE is a deep learning method to analyze chromatin conformations obtained from single-cell imaging studies. 

ChromVAE has two major contributions:

* The latent space obtained from chromVAE monitors the progression of chromatin folding process.

* The probability estimated from chromVAE provides chromatin energy landscape.

More details can be found in the paper [Characterizing Chromatin Folding Coordinate and Landscape with Deep Learning](https://www.biorxiv.org/content/10.1101/824417v1) by Wen Jun Xie, Yifeng Qi and Bin Zhang.

## Requirements
The package has been tested on CentOS Linux release 7.6 with the following software:
Python 3.7, PyTorch 1.2, Numpy 1.16, Pickle 4.0

## Dataset
The chromatin imaging data were downloaded from [Bintu et. al., Science, 2018, 362, eaau1783](https://github.com/BogdanBintu/ChromatinImaging). The distance map was then binarized to contact map provided in the [`./data/`](./data) directory. 90-kb resolution was used and there are 378 chromatin contacts for the studies region.

* [`./data/HCT116/`](./data/HCT116/): contact matrixes for wild-type cell

* [`./data/HCT116_auxin/`](./data/HCT116_auxin/): contact matrixes for cohesin-depleted cell

## Source code 
* [`./script/VAE_combine_train.py`](./script/VAE_combine_train.py):  code used to train chromVAE

* [`./script/VAE_combine_latent.py`](./script/VAE_combine_latent.py):  code used to get the latent space after training chromVAE

## Analysis
We also include the code to analyze the latent space in the [`./analysis/analysis_latent.ipynb`](./analysis/analysis_latent.ipynb).
