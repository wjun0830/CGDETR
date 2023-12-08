import os
import pdb
import json
import nncore

meta = nncore.load('./youtube_anno.json')

slowfast_list = os.listdir('../../../features/youtube_uni/vid_slowfast')
clip_list = os.listdir('../../../features/youtube_uni/vid_clip')
final_list = set(slowfast_list).intersection(set(clip_list))
final_list = [x[:-4] for x in list(final_list)]

save_dict = {}

for k, v in meta.items():
    if k in final_list:
        save_dict[k] = v

with open('./youtube_train.json', 'w') as json_file:
        json_file.write(json.dumps(save_dict))