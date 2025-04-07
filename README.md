# Multimodal Tumour Type Classification using Deep Learning

AThis project investigates ways to represent and embed somatic mutation data for tumour type and subtype classification with attention-based deep learning model Mutation-Attention (MuAT). Master’s thesis, with the title Multimodal Tumour Type and Subtype Classification using Deep Learning, related to this project can be read on [Aalto Doc](https://urn.fi/URN:NBN:fi:aalto-202503172818) where the methods, data and results are explained in detail. 

## Introduction 

Tumour types in different tissues are characterised by distinct patterns of somatic mutations because tissues exposure to different environmental factors. This makes classification of the tumour types possible from these patterns. In this project classification is done utilising modified MuAt model. Original MuAt model utilises trinucleotide mutation motifs, genomic position and GES (genic, exonic, and strand orientation) annotation in tumour type classification, and it is developed by Prima Sanjaya, and the model can be found on this repository. 

In this project, difference compared to original MuAt model is that somatic mutations have more flanking bases around them, i.e., instead of trinucleotide motifs the longer motifs are utilised. Additionally, these motifs are embedded utilising DNABERT-2 model. Another embedding method is single base one-hot encoding the sequence which is embed with linear layer. By this way, it is researched if longer sequence context gives additional information on tumour classification.  

Furthermore, chromatin states are utilised as a new data attribute to test if they enhance the classification accuracy. Chromatin states are embedded utilising unpublished model called EpicVAE developed by Katri Maljanen. 

## Content of Repository 

- `configs/` - configuration files for the model training    
- `data/` - data utilised for the training    
- `ensemble/` - ensemble models and the logits     
- `environment/` - programming environment scripts   
- `extfiles/` - help files utilised in training and data processing 
- `figures/` - saved training figures
- `notebooks/` - jupyter notebooks utilised in data visualisation and exploring data processing
- `slurm/` - slurm scripts utilised to run codes on cluster  
- `src/` - source codes
- `Test_MuAt`- test original MuAt model
- `tests/` - tests for checking preprocessing results 


The best trained model checkpoints can be downloaded [here](https://docs.google.com/uc?export=download&id=1ImgeC_d0A5xcFCD24-zQ51JwS8IDu7N7)
The UMAP models and data utilised for these models can be downloaded [here](https://docs.google.com/uc?export=/1hvwyC16iDkNwbIHxHq-UIPpV5EqrW_41)

## Environment 

```
conda create -n "env_name" python=3.8 -y
conda activate "env_name"
pip3 install -r ./environment/dna_requirements.txt
```

| package | version |
| ------- | ------- |
| accelerate | 0.23.0 |
| antlr4-python3-runtime | 4.9.3 |
| einops | 0.6.1 |
| evaluate | 0.4.0 |
| dask | 2023.5.0 |
| hugginface-hub | 0.17.3 |
| natsort | 8.4.0 |
| numpy | 1.23.0 |
| notebook | 7.2.2 |
| omegaconf | 2.3.0 |
| pandas | 2.0.3 |
| peft | 0.5.0 |
| pytorch | 1.13.1+cu11.8 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.2 |
| safetensors | 0.3.3 |
| tensorboard |2.14.0 |
| triton | 2.0.0.dev20221102 |
| transformers | 4.29.2 |
| umap-learn | 0.5.6 |

***Note: Triton is not utilised in this installation***

### DNABERT-2 & Triton

I tried to follow the set-up instructions in [the githu repository of the DNABERT](https://github.com/MAGICS-LAB/DNABERT_2?tab=readme-ov-file#3-setup-environment) but the building of triton library from source fails on MLBioMed group's cluster even when following the tips in [OpenAI's github repo](https://github.com/triton-lang/triton). Instead, I created requirements.txt file containing correct versions of the packages. **Note: use python version 3.8**.

**UPDATE 13.8.2024**: For some reason needed version of triton cannot be installed anymore. I created issue about the problem, and it can be followed [here](https://github.com/triton-lang/triton/issues/4511)

**UPDATE 28.3.2025**: Triton version 2.0.0.dev20221102 is not available anymore via pip. It is possible to use DNABERT without Flash Attention meaning that there is no need to install Triton. If you want to use flash Attention, you need to build the Triton from source or use legacy version of Triton. Instructions to install legacy version can be found [here](https://github.com/triton-lang/triton/issues/4511#issuecomment-2620936023) 

## How to Process Data

Data can be preprocessed using `annotate_mutations.py` and `embedd_mutations.py` modules. 

`annotate_mutations.py` module orders the mutations in csv files and replaces the mutation tokens with actual sequences. Module uses following command line arguments:

- `-v` or `--verbose` - flag to increase output verbosity
- `-o` or `--muat_orig` - flag to save sequences with the mutation tokens, default: False
- `-l` or `--length` - length of the mutation motif, default value: 3
- `-c` or `--continue_from` - the index of the file where to continue annotation, optional, default value: -1, meaning no continuation index/ start from the beginning 
- `-e` or `--end` - the index of the file where to end, optional, default value: -1, meaning no ending index
- `predict_filepath` - full path to csv file containing the data files to handle
- `tmp_dir` - path to directory where to store the processed files

Example run could look like this:

```
python3 ./src/annotate_mutations.py --verbose --length 7 --continue_from 2500 /csc/epitkane/projects/multimodal/data/utils/data_preprocessing_true.tsv.gz /csc/epitkane/projects/multimodal/data/DNABERT_motif3
```


## How to Train Model

Training and validating the model happens utilising `main.py` module. Following arguments are needed to for model training:

- `--config_file` - path to configuration file utilised for the training
- `--input` - path to data directory (this parameter is optional if you provide the data path in configuration file), created for using in csc Mahti, optional
- `--fold` - fold of the cross validation, optional
- `--load` - path to checkpoint where to continue the training of the model, optional
- `--valid` - Flag that marks if the model is utilised to make prediction or validation from the data, default: False
- `--full_data` - Flag that marks if the whole dataset is utilised in prediction without any split, default value: False

To train or use the model the config file is needed. The config files utilised for training aresaved in `configs/` and they could be utilised as an example. Parameters of the config file are explained in `configs/config_parameters.md`

Example run could look like this:

```
python3 ./src/main.py --config_file ./configs/config_motif3.ini
```
You can find slurm scripts utilised for training models in FIMM cluster and in CSC MAHti in `slurm/run_train.sh` and `slurm/mahti/`

## RAW DATA
https://docs.icgc-argo.org/docs/data-access/icgc-25k-data

The instructions for downloading RAW data and handling it, can be found in github repository of [the original MuAt model](https://github.com/primasanjaya/muat-github/blob/master/README_training.md). 

## Authors 
Antti Huttunen


