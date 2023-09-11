# Introduction
GPSite is a geometry-aware multi-task network for simultaneously predicting binding residues of DNA, RNA, peptide, protein, ATP, HEM, and metal ions on proteins. By leveraging the informative sequence embeddings and predicted structures from pre-trained language models, GPSite is liberated from the reliance on MSA or experimental protein structures. GPSite is easy to install and run, and is also fast and accurate (surpassing state-of-the-art sequence-based and structure-based methods). If your input is small, you are recommended to use our [GPSite webserver](https://bio-web1.nscc-gz.cn/app/GPSite). We also provide the binding site annotations for Swiss-Prot in our [GPSiteDB](https://bio-web1.nscc-gz.cn/database/GPSiteDB/).
![workflow](https://github.com/biomed-AI/GPSite/blob/main/image/workflow.jpg)

# System requirement
GPSite is mainly based on the following packages:  
- python  3.8.16  
- numpy  1.24.3  
- pytorch  1.13.1  
- pytorch-scatter  2.1.1  
- pytorch-cluster  1.6.1  
- pyg  2.3.0  
- biopython  1.81  
- fair-esm  2.0.0  
- dllogger  1.0.0  
- openfold  1.0.1  
- sentencepiece  0.1.99  
- transformers  4.30.1  
While we have not tested other versions, any reasonably recent versions of these requirements should work.

# Install and set up GPSite
**1.** Clone this repository by `git clone https://github.com/biomed-AI/GPSite.git` or download the code in ZIP archive
**2.** Install the packages required by GPSite. To install [ESMFold](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans), one can follow their official tutorials. However, at the time of writing, we found some commands in the installation tutorial of ESMFold didn't work. To avoid unnecessary troubles, you can install GPSite according to the following instructions that we cleaned up:  
**a.** To use the ESMFold model, make sure you start from an environment with python <= 3.9 and pytorch installed. Here we use python 3.8.16 and pytorch 1.13.1 with cuda version of 11.6. Then run:
```
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install modelcif==0.7
```
Finally, download the [openfold](https://github.com/aqlaboratory/openfold) ZIP archive, unzip it, and run `python3 setup.py install`.  
**b.** To use the ProtTrans model, run:
```
pip install SentencePiece transformers
```
**c.** Finally, install the remaining packages required by GPSite:
```
conda install pyg -c pyg
conda install pytorch-scatter -c pyg
conda install pytorch-cluster -c pyg
```
**3.** Download the pre-trained ProtT5-XL-UniRef50 model in [here](https://zenodo.org/record/4644188) (~ 5.3 GB). The ESM2 and ESMFold models will be automatically downloaded at the first time you run GPSite, which will take some time (~ 7.9 GB).  
**4.** Set the path variable `ProtTrans_path` in `./script/predict.py`  
**5.** Add permission to execute for DSSP by `chmod +x ./script/feature_extraction/mkdssp`  

# Run GPSite for prediction
Simply run the following command to predict the binding sites of the sequences in `demo.fa` on GPU (id = 0):
```
python ./script/predict.py -i ./example/demo.fa -o ./example/ --gpu 0
```
Omitting the `--gpu` parameter will make GPSite run on CPU, which will take more time especially for the structure prediction pipeline. The prediction results will be saved under `./example/demo/pred/`. Here we provide the corresponding canonical input and prediction results under `./example/` for your reference. Residues with predicted binding scores > 0.5 should be considered binding sites.

# Dataset and model
We provide the datasets and the trained models here for those interested in reproducing our paper.  
The protein binding site datasets used in this study are stored in `./datasets/`.  
The trained GPSite models can be found under `./model/`.

# Citation and contact
Citation:  

Contact:  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
