CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port=8245 baseline.py \
    --options \
    exp_name navie_internvl \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/internvl_brief.json \
    mode ours_baseline