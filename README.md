# OKGIT
Source code of the proposed models and experiments in the paper "[OKGIT: Open Knowledge Graph Link Prediction with Implicit Types]()" to be presented in the Findings of ACL 2021.

![](https://github.com/chandrahasd/OKGIT/blob/main/arch_okgit_camera_ready.png)

## Description
  The code is based on [CaRE](https://github.com/malllabiisc/CaRE) and [LAMA](https://github.com/facebookresearch/LAMA). The datasets used in this work can be found [here](https://github.com/malllabiisc/CaRE/tree/master/Data). The pre-trained language models (BERT-base, BERT-large, RoBERTa) used in the paper can be found [here](https://huggingface.co/models).



## Installation
  Clone the repository
  
  ```bit
  git clone "https://github.com/Chandrahasd/OKGIT.git"
  ```

  Install dependencies
  
  ```bash
  pip install -r requirements.txt
  ```



## Running
  #### ReVerb20K
  
  ```python
  python src/main.py --dataset ReVerb20K --n_epochs 500 --model_name OKGIT --nfeats 300 --lm bert --reverse --type-loss mse --type_transform identity --type-dim 300 --type-weight 0.01 --name testrun --bmn bert-large-uncased --type_composition add --type_composition_weight 5.0 --gpu 0 --nocomet
  ```
  
  #### ReVerb45K
  
  ```python
  python src/main.py --dataset ReVerb45K --n_epochs 500 --model_name OKGIT --nfeats 300 --lm bert --reverse --type-loss mse --type_transform identity --type-dim 100 --type-weight 0.0 --name testrun --bmn bert-large-uncased --type_composition add --type_composition_weight 2.0 --gpu 0 --nocomet
  ```
  
  #### ReVerb20KF
  
  ```python
  python src/main.py --dataset ReVerb20KF --n_epochs 500 --model_name OKGIT --nfeats 300 --lm bert --reverse --type-loss mse --type_transform identity --type-dim 300 --type-weight 0.001 --name testrun --bmn bert-base-uncased --type_composition add --type_composition_weight 5.0 --gpu 0 --nocomet
  ```
  
  #### ReVerb45KF
  
  ```python
  python src/main.py --dataset ReVerb45KF --n_epochs 500 --model_name OKGIT --nfeats 300 --lm bert --reverse --type-loss mse --type_transform identity --type-dim 300 --type-weight 0.001 --name testrun --bmn bert-base-uncased --type_composition add --type_composition_weight 0.25 --gpu 0 --nocomet
  ```



  ## Generating Single Token Datasets
  ```
  python src/preprocess/filter_triples.py --dataset <path-to-source-dataset> --bert-vocab <path-to-bert-vocab-file> 
  ```



  ## Reference
  The OKGIT model is described in the following paper:
  
  ```bibtex
    @inproceedings{chandrahas-talukdar-2021-okgit,
    title = "{OKGIT}: {O}pen Knowledge Graph Link Prediction with Implicit Types",
    author = "Chandrahas, .  and
      Talukdar, Partha",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.225",
    doi = "10.18653/v1/2021.findings-acl.225",
    pages = "2546--2559",
    }
  ```
