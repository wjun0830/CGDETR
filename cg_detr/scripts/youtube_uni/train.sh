dset_name=youtube_uni
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_youtubeuni
exp_id=exp


######## data paths
# train_path=data/youtube_uni/youtube_train.jsonl
# eval_path=data/youtube_uni/youtube_anno.jsonl
train_path=data/youtube_uni/youtube_train.jsonl
eval_path=data/youtube_uni/youtube_valid.jsonl
eval_split_name=val

######## setup video+text features
# feat_root=../features/tvsum
feat_root=../features/youtube_uni

# # video features
v_feat_dim=2816
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/vid_clip)
v_feat_dirs+=(${feat_root}/vid_slowfast)

# # text features
t_feat_dir=${feat_root}/txt_clip/ # maybe not used
t_feat_dim=512


#### training
bsz=4
lr=2e-4
enc_layers=3
dec_layers=3
t2v_layers=2
moment_layers=1
dummy_layers=2
sent_layers=1

for num_dummies in 1
do 
    for seed in 2018
    do 
        for dset_domain in dog gymnastics parkour skating skiing surfing
        do
            for num_prompts in 1 2
            do
                PYTHONPATH=$PYTHONPATH:. python cg_detr/train.py \
                --dset_name ${dset_name} \
                --ctx_mode ${ctx_mode} \
                --train_path ${train_path} \
                --eval_path ${eval_path} \
                --eval_split_name ${eval_split_name} \
                --v_feat_dirs ${v_feat_dirs[@]} \
                --v_feat_dim ${v_feat_dim} \
                --t_feat_dir ${t_feat_dir} \
                --t_feat_dim ${t_feat_dim} \
                --bsz ${bsz} \
                --results_root ${results_root}/${dset_domain} \
                --exp_id ${exp_id} \
                --max_v_l 1000 \
                --n_epoch 1000 \
                --lr_drop 2000 \
                --max_es_cnt -1 \
                --seed $seed \
                --lr ${lr} \
                --dset_domain ${dset_domain} \
                --enc_layers ${enc_layers} \
                --dec_layers ${dec_layers} \
                --t2v_layers ${t2v_layers} \
                --moment_layers ${moment_layers} \
                --dummy_layers ${dummy_layers} \
                --sent_layers ${sent_layers} \
                --clip_length 1 \
                --lw_saliency 4 \
                --num_dummies ${num_dummies} \
                --num_prompts ${num_prompts} \
                --total_prompts 10 \
                --num_workers 4
                ${@:1}
            done
        done
    done
done
