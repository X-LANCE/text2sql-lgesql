# LGESQL

This is the project containing source code for the paper [*LGESQL: Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations*](https://arxiv.org/abs/2004.12299) in **ACL 2021 main conference**. If you find it useful, please cite our work.

    @article{cao2021lgesql,
        title={LGESQL: Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations},
        author={Cao, Ruisheng and Chen, Lu and Chen, Zhi and Zhu, Su and Yu, Kai},
        journal={arXiv preprint arXiv:2106.01093},
        year={2021}
    }


## Create environment and download dependencies

1. Firstly, create conda environment `text2sql`:
    
        conda create -n text2sql python=3.6
        source activate text2sql
        pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

2. Next, download dependencies:


