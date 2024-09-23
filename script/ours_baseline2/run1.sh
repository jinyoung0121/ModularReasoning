CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node 1 --master_port=1111 ours_baseline2.py \
    --options \
    exp_name internvl_internlm \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 5 \
    log_freq 5 \
    video_context datas/internvl_brief.json \
    mode ours_baseline