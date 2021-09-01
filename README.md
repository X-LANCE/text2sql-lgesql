# LGESQL

This is the project containing source code for the paper [*LGESQL: Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations*](https://arxiv.org/abs/2004.12299) in **ACL 2021 main conference**. If you find it useful, please cite our work.

        @inproceedings{cao-etal-2021-lgesql,
                title = "{LGESQL}: Line Graph Enhanced Text-to-{SQL} Model with Mixed Local and Non-Local Relations",
                author = "Cao, Ruisheng  and
                Chen, Lu  and
                Chen, Zhi  and
                Zhao, Yanbin  and
                Zhu, Su  and
                Yu, Kai",
                booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
                month = aug,
                year = "2021",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.acl-long.198",
                doi = "10.18653/v1/2021.acl-long.198",
                pages = "2541--2555",
        }


## Create environment and download dependencies
The following commands are provided in `setup.sh`.

1. Firstly, create conda environment `text2sql`:
  - In our experiments, we use **torch==1.6.0** and **dgl==0.5.3** with CUDA version 10.1
  - We use one GeForce RTX 2080 Ti for GLOVE and base-series pre-trained language model~(PLM) experiments, one Tesla V100-PCIE-32GB for large-series PLM experiments
    
        conda create -n text2sql python=3.6
        source activate text2sql
        pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

2. Next, download dependencies:

        python -c "import stanza; stanza.download('en')"
        python -c "from embeddings import GloveEmbedding; emb = GloveEmbedding('common_crawl_48', d_emb=300)"
        python -c "import nltk; nltk.download('stopwords')"

3. Download pre-trained language models from [`Hugging Face Model Hub`](https://huggingface.co/models), such as `bert-large-whole-word-masking` and `electra-large-discriminator`, into the `pretrained_models` directory. The vocab file for [`glove.42B.300d`](http://nlp.stanford.edu/data/glove.42B.300d.zip) is also pulled: (please ensure that `Git LFS` is installed)

        mkdir -p pretrained_models && cd pretrained_models
        git lfs install
        git clone https://huggingface.co/bert-large-uncased-whole-word-masking
        git clone https://huggingface.co/google/electra-large-discriminator
        mkdir -p glove.42b.300d && cd glove.42b.300d
        wget -c http://nlp.stanford.edu/data/glove.42B.300d.zip && unzip glove.42B.300d.zip
        awk -v FS=' ' '{print $1}' glove.42B.300d.txt > vocab_glove.txt

## Download and preprocess dataset

1. Download, unzip and rename the [spider.zip](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0) into the directory `data`.

2. Merge the `data/train_spider.json` and `data/train_others.json` into one single dataset `data/train.json`.

3. Preprocess the train and dev dataset, including input normalization, schema linking, graph construction and output actions generation. (Our preprocessed dataset can be downloaded [here](https://drive.google.com/file/d/1L8sWlp7J9LWjw9MP2bHGsf0wC4xLAyxO/view?usp=sharing))

        ./run/run_preprocessing.sh

## Training

Training LGESQL models with GLOVE, BERT and ELECTRA respectively:
  - msde: mixed static and dynamic embeddings
  - mmc: multi-head multi-view concatenation


        ./run/run_lgesql_glove.sh [mmc|msde]
        ./run/run_lgesql_plm.sh [mmc|msde] bert-large-uncased-whole-word-masking
        ./run/run_lgesql_plm.sh [mmc|msde] electra-large-discriminator

## Evaluation and submission

1. Create the directory `saved_models`, save the trained model and its configuration (at least containing `model.bin` and `params.json`) into a new directory under `saved_models`, e.g. `saved_models/electra-msde-75.1/`.

2. For evaluation, see `run/run_evaluation.sh` and `run/run_submission.sh` (eval from scratch) for reference.

3. Model instances and submission scripts are available in [codalab:plm](https://worksheets.codalab.org/worksheets/0x53017948b7dc4cbd95d3191a35f6b6b2) and [google drive](https://drive.google.com/file/d/1ALf5ycxMViHrT5WGuFO3g9eT7R2S1rgy/view?usp=sharing): including submitted BERT and ELECTRA models. Codes and model for GLOVE are deprecated.


## Results
Dev and test **EXACT MATCH ACC** in the official [leaderboard](https://yale-lily.github.io//spider), also provided in the `results` directory:

| model | dev acc | test acc |
| :---: | :---: | :---: |
| LGESQL + GLOVE | 67.6 | 62.8 |
| LGESQL + BERT | 74.1 | 68.3 |
| LGESQL + ELECTRA | 75.1 | 72.0 |

## Acknowledgements

We would like to thank Tao Yu, Yusen Zhang and Bo Pang for running evaluations on our submitted models. We are also grateful to the flexible semantic parser [TranX](https://github.com/pcyin/tranX) that inspires our works.
