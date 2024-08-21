# CG-DETR : Calibrating the Query-Dependency of Video Representation via Correlation-guided Attention for Video Temporal Grounding
 Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding
> WonJun Moon, Sangeek Hyun, SuBeen Lee, Jae-Pil Heo <br>
> Sungkyunkwan University

##### [Arxiv](https://arxiv.org/abs/2311.08835)


🥇[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/correlation-guided-query-dependency/highlight-detection-on-qvhighlights)](https://paperswithcode.com/sota/highlight-detection-on-qvhighlights?p=correlation-guided-query-dependency)<br>
🥇[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/correlation-guided-query-dependency/moment-retrieval-on-qvhighlights)](https://paperswithcode.com/sota/moment-retrieval-on-qvhighlights?p=correlation-guided-query-dependency)<br>
🥇[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/correlation-guided-query-dependency/highlight-detection-on-tvsum)](https://paperswithcode.com/sota/highlight-detection-on-tvsum?p=correlation-guided-query-dependency)<br>
🥇[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/correlation-guided-query-dependency/highlight-detection-on-youtube-highlights)](https://paperswithcode.com/sota/highlight-detection-on-youtube-highlights?p=correlation-guided-query-dependency)<br>
🥇[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/correlation-guided-query-dependency/natural-language-moment-retrieval-on-tacos)](https://paperswithcode.com/sota/natural-language-moment-retrieval-on-tacos?p=correlation-guided-query-dependency)<br>
🥇[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/correlation-guided-query-dependency/moment-retrieval-on-charades-sta)](https://paperswithcode.com/sota/moment-retrieval-on-charades-sta?p=correlation-guided-query-dependency)

<p align="center">
 <img src="https://github.com/wjun0830/CGDETR/assets/31557552/1aa24ace-3aa9-452b-ac13-1798467da10a" width="80%">
</p>

### 🔖 Abstract
Recent endeavors in video temporal grounding enforce strong cross-modal interactions through attention mechanisms to overcome the modality gap between video and text query.
However, previous works treat all video clips equally regardless of their semantic relevance with the text query in attention modules.
In this paper, our goal is to provide clues for query-associated video clips within the crossmodal encoding process.
With our Correlation-Guided Detection Transformer~(CG-DETR), we explore the appropriate clip-wise degree of cross-modal interactions and how to exploit such degrees for prediction.
First, we design an adaptive cross-attention layer with dummy tokens. 
Dummy tokens conditioned by text query take a portion of the attention weights, preventing irrelevant video clips from being represented by the text query.
Yet, not all word tokens equally inherit the text query's correlation to video clips. 
Thus, we further guide the cross-attention map by inferring the fine-grained correlation between video clips and words. 
We enable this by learning a joint embedding space for high-level concepts, \textit{i.e}., moment and sentence level, and inferring the clip-word correlation.
Lastly, we use a moment-adaptive saliency detector to exploit each video clip's degrees of text engagement.
We validate the superiority of CG-DETR with the state-of-the-art results on various benchmarks for both moment retrieval and highlight detection.

----------
## 📢 To be updated
### Todo
- [x] : Upload instruction for dataset download
- [x] : Update model zoo
- [x] : Upload implementation

----------

## 📑 Datasets
<b>QVHighlights</b> : Download official feature files for QVHighlights dataset from [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB). 
```
tar -xf path/to/moment_detr_features.tar.gz
```
If inaccessible, then download from
> <b> [QVHighlight](https://drive.google.com/file/d/1LXsZZBsv6Xbg_MmNQOezw0QYKccjcOkP/view?usp=sharing)</b> 9.34GB. <br>

For other datasets, we provide extracted features:

> <b> [Charades-STA](https://drive.google.com/file/d/1B2721QC799qbbGLGSa7DkXJjdRefvZf-/view?usp=sharing )</b> 33.18GB. (Including SF+C and VGG features) <br>
> <b> [TACoS](https://drive.google.com/file/d/1_IaKMjKw3nNaSsvN28ZucfM4K-ivZTHw/view?usp=sharing) </b> 290.7MB. <br>
> <b> [TVSum](https://drive.google.com/file/d/10Ji9MrlDK_4FdD3HotrVc407xVr4arsL/view) </b> 69.1MB. <br>
> <b> [Youtube](https://drive.google.com/file/d/1qVhb33ABnWqiHjT22f54fKhSlf2Z-z0f/view?usp=sharing) </b> 191.7MB. <br>

After downloading, either prepare the data directory as below or change 'feat_root' in TVSum shell files under 'cg_detr/scripts/*/'.

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
    └── qvhighlight
    └── charades
    └── tacos
    └── tvsum
    └── youtube_uni

```


## 🛠️ Installation
Python version 3.7 is required.
1. Clone this repository.
```
git clone https://github.com/wjun0830/CGDETR.git
```
2. Download the packages we used for training.
```
pip install -r requirements.txt
```

## 🚀 Training
We provide training scripts for all datasets in `cg_detr/scripts/` directory.


### QVHighlights Training
Training can be executed by running the shell below:
```
bash cg_detr/scripts/train.sh  
```
Best validation accuracy is yielded at the last epoch. 

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

### QVHighlights w/ Pretraining Training
Training can be executed by running the shell below:
```
bash cg_detr/scripts/train.sh --num_dummies 45 --num_prompts 1 --total_prompts 10 --max_q_l 75 --resume pt_checkpoints/model_e0009.ckpt --seed 2018
```
Checkpoints for pretrained checkpoint 'model_e0009.ckpt' is available [here](https://drive.google.com/drive/folders/1iH4Jfg_5rDA-N1nkg_iqRk-mIcmQblQW?usp=sharing).

## 👀 QVHighlights Evaluation and Codalab Submission
Once the model is trained, `hl_val_submission.jsonl` and `hl_test_submission.jsonl` can be yielded by running inference.sh.
Compress them into a single `.zip` file and submit the results.
```
bash cg_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash cg_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```
where `direc` is the path to the saved checkpoint.
For more details, check [standalone_eval/README.md](standalone_eval/README.md).

## 📹 Others (Custom video inference / training)
- Running predictions on customized datasets is also available.
Note that only the CLIP-only trained model is available for custom video inference. <br>
You can either <br>
  1)`Preparing your custom video and text query under 'run_on_video/example',` <br>
  2)`Modify the youtube video url and custom text query in 'run_on_video/run.py'` <br>
  (youtube_url : video link url, [vid_st_sec, vid_ec_sec] : start and end time of the video (specify less than 150 frames), desired_query : text query) <br>
Then, run the following commands:`
```
pip install ffmpeg-python ftfy regex
PYTHONPATH=$PYTHONPATH:. python run_on_video/run.py
```

- For instructions for training on custom datasets, check [here](https://github.com/jayleicn/moment_detr).


## 📦 Model Zoo 
Dataset | Model file
 -- | -- 
QVHighlights | [checkpoints](https://drive.google.com/drive/folders/1_hEqXbvDv4AyEn5unyn_kE784ruqrzEJ?usp=sharing)
Charades (Slowfast + CLIP) | [checkpoints](https://drive.google.com/drive/folders/1x937GAd8brWhWy6_GGXl6QYN4bVtz7BN?usp=sharing)
Charades (VGG) | [checkpoints](https://drive.google.com/drive/folders/1UEwcuVYLjCLmeJWM-ZpfXZQ8PMoluAGU?usp=sharing)
TACoS | [checkpoints](https://drive.google.com/drive/folders/1r6sB-9KPf5awkhmx-iPwjj_B24i1t7OY?usp=sharing)
TVSum | [checkpoints](https://drive.google.com/drive/folders/1RXZxpe__tUidoiP4FWJuZVQy84iwKyti?usp=sharing)
Youtube-HL | [checkpoints](https://drive.google.com/drive/folders/1Mbri6RVb9W31gLfpvQlGXGasg5SPiet2?usp=sharing)
QVHighlights w/ PT (47.97 mAP) | [checkpoints](https://drive.google.com/drive/folders/1iH4Jfg_5rDA-N1nkg_iqRk-mIcmQblQW?usp=sharing)
QVHighlights only CLIP | [checkpoints](https://drive.google.com/drive/folders/1LMMY349TKR7wlAEqHxlrsEGULHRIN21o?usp=sharing)
 
## 📖 BibTeX 
If you find the repository or the paper useful, please use the following entry for citation.
```
@article{moon2023correlation,
  title={Correlation-guided Query-Dependency Calibration in Video Representation Learning for Temporal Grounding},
  author={Moon, WonJun and Hyun, Sangeek and Lee, SuBeen and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2311.08835},
  year={2023}
}
```

## ☎️ Contributors and Contact
If there are any questions, feel free to contact the authors: WonJun Moon (wjun0830@gmail.com), Sangeek Hyun (hse1032@gmail.com), and SuBeen Lee (leesb7426@gmail.com)

## ☑️ LICENSE
The annotation files and many parts of the implementations are borrowed from [Moment-DETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.
 
