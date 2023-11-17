# CG-DETR : Calibrating the Query-Dependency of Video Representation via Correlation-guided Attention for Video Temporal Grounding
by 
WonJun Moon, SangEek Hyun, SuBeen Lee, Jae-Pil Heo

Sungkyunkwan University

[[Arxiv](https://arxiv.org/abs/2303.13874)]

----------
## To be updated
### Todo
- [ ] : Upload implementation
- [ ] : Upload instruction for dataset download
- [ ] : Update results
- [ ] : Update model zoo


## Prerequisites
<b>1. Prepare datasets</b>

<b>QVHighlights</b> : Download official feature files for QVHighlights dataset from Moment-DETR. 

Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB).
```
tar -xf path/to/moment_detr_features.tar.gz
```

<b> Charades-STA </b> : TBD

<b> TACoS </b> : TBD

<b>TVSum</b> : Download feature files for TVSum dataset from [TVSum](https://drive.google.com/file/d/10Ji9MrlDK_4FdD3HotrVc407xVr4arsL/view) (69.1MB),
and either extract it under '../features/tvsum/' directory or change 'feat_root' in TVSum shell files under 'cg_detr/scripts/tvsum/'.
Link from UMT is broken. TBD

<b>Youtube</b> : TBD

After downloading, prepare the data directory as below:
Else, you can change the data directory by modifying 'feat_root' in shell scripts under 'cg_detr/scripts/' directory.
```txt
.
├── CGDETR
│   ├── cg_detr
│   └── data
│   └── results
│   └── run_on_video
│   └── standalone_eval
│   └── utils
├── features
│   └── qvhighlight
│   └── charades
│   └── tacos
│   └── tvsum
│   └── youtube_uni

```




<b>2. Install dependencies.</b>
Python version 3.7 is required.
```
pip install -r requirements.txt
```

## Training & Evaluation
We provide training scripts for all datasets in `cg_detr/scripts/` directory.

### QVHighlights
### Training
Training can be executed by running the shell below:
```
bash cg_detr/scripts/train.sh  
```
Best validation accuracy is yielded at the last epoch. 

### Inference Evaluation and Codalab Submission for QVHighlights
Once the model is trained, `hl_val_submission.jsonl` and `hl_test_submission.jsonl` can be yielded by running inference.sh.
```
bash cg_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash cg_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```
where `direc` is the path to the saved checkpoint.
For more details for submission, check [standalone_eval/README.md](standalone_eval/README.md).




### Charades-STA
For training, run the shell below:
```
bash cg_detr/scripts/charades_sta/train.sh
bash cg_detr/scripts/charades_sta/train_vgg.sh  
```

### TACoS
For training, run the shell below:
```
bash cg_detr/scripts/tacos/train.sh  
```

### TVSum
For training, run the shell below:
```
bash cg_detr/scripts/tvsum/train_tvsum.sh  
```
Best results are stored in 'results_[domain_name]/best_metric.jsonl'.


### Youtube-hl
For training, run the shell below:
```
bash cg_detr/scripts/youtube_uni/train.sh  
```
Best results are stored in 'results_[domain_name]/best_metric.jsonl'.

### Others
- Runninng predictions on customized datasets is also available as we use the official implementation for Moment-DETR / QD-DETR as the codebase.
Note that only the CLIP-only trained model is available for custom video inference.
Once done `Preparing your custom video and text query under 'run_on_video/example',
run the following commands:`
```
pip install ffmpeg-python ftfy regex
PYTHONPATH=$PYTHONPATH:. python run_on_video/run.py
```

For instructions for training on custom datasets, check [here](https://github.com/jayleicn/moment_detr).


## Model Zoo 
Dataset | Model file
 -- | -- 
QVHighlights (Slowfast + CLIP) | 
Charades (Slowfast + CLIP) | 
TACoS | 
Youtube-HL | 
YOutube-HL | 
 
## BibTeX 
 

If you find the repository or the paper useful, please use the following entry for citation.
```

```

## Contributors and Contact


## LICENSE
The annotation files and many parts of the implementations are borrowed [Moment-DETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.
 
