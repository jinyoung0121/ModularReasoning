CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port=1103 ours_baseline2.py \
    --options \
    exp_name internvl_internlm \
    dataset.dataset_name NExTGQA \
    dataset.data_path datas/NExT-GQA \
    dataset.video_path datas/NExT-QA/ \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/internvl_brief.json \
    mode ours_baseline