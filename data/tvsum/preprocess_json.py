import json
from tvsum_splits import TVSUM_SPLITS

from glob import glob
import os

if __name__ == "__main__":
    
    anno_path = "tvsum_train_sfc.json"
    
    train_path = "tvsum_train_sfc.jsonl"
    valid_path = "tvsum_val_sfc.jsonl"
    
    feat_path = "/dataset/MRHD/features/tvsum_sfc"
    feat_dirs = glob(os.path.join(feat_path, "*"))

    # print(feat_dirs)
    # get valid videos, what we have actual features
    available_feat = []
    for fd in feat_dirs:
        fns = glob(os.path.join(fd, "*"))
        # print(fns)
        cur_available_feat = []
        for fn in fns:
            # print(fn)
            fn = os.path.basename(fn[:-4])
            # print(fn)
            cur_available_feat.append(fn)
        available_feat.append(cur_available_feat)
    # print(available_feat)
    # print(len(available_feat))

    available_fns = []
    for fn in available_feat[0]:
        if fn not in available_feat[1]:
            continue
        if fn not in available_feat[2]:
            continue
        available_fns.append(fn)

    
    with open(anno_path, "r") as f:
        data = json.load(f)
        
    # process data
    
    # TVsum saves below keywords
    # [qid, query, duration, vid, relevant_clip_ids, relevant_windows, label, domain]
    
    anno_train = []
    anno_valid = []
    
    for k in data.keys():
        # print(k)
        if k not in available_fns:
            print(f"There is no features for video {k}")
            continue
        
        qid = k
        query = data[k]['title']
        duration = float(data[k]['frames']) / float(data[k]['fps']) # saving the number of frames
        vid = k # In dataloader, meta data is retrieved by vid name, so we save vid as key
        relevant_clip_ids = None # do not save MR-relevant features
        relevant_windows = None # do not save MR-relevant features
        
        # UniVTG only regards match > 0 as salience clips
        # https://github.com/showlab/UniVTG/blob/main/main/dataset.py, 846-848 lines
        import numpy as np
        print(np.array(data[k]['anno']).shape)
        print(duration)
        saliency = np.array(data[k]['anno']).sum(1).tolist()
        # saliency = [1 if s > 0 else 0 for s in data[k]['match']]
        
        label = [[i] for i in saliency]  # following tvsum format
        domain = data[k]['domain']
        
        # save keywords of youtube-hl
        # frames = float(data[k]['frames'])
        # fps = float(data[k]['fps'])
        # clip = data[k]['clip']
        # match = data[k]['match']
        
        
        
        cur_data = {
            'qid': qid,
            'query': query,
            'duration': duration,
            'vid': vid,
            'relevant_clip_ids': relevant_clip_ids,
            'relevant_windows': relevant_windows,
            'label': label,
            'domain': domain,
        }
        
        if cur_data['vid'] in TVSUM_SPLITS[domain]['train']:
            anno_train.append(cur_data)
        elif cur_data['vid'] in TVSUM_SPLITS[domain]['val']:
            anno_valid.append(cur_data)
        else:
            print(f"No anno for {cur_data['vid']}")
            # print(k)
            
    
    print('# of total data:', len(data))
    print('# of train data:', len(anno_train))
    print('# of valid data:', len(anno_valid))
    
    with open(train_path, "w") as f:
        for cur_anno in anno_train:
            json.dump(cur_anno, f)
            f.write("\n")
    
    with open(valid_path, "w") as f:
        for cur_anno in anno_valid:
            json.dump(cur_anno, f)
            f.write("\n")
        