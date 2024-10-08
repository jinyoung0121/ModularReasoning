CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node 2 --master_port=8245 baseline.py \
    --options \
    exp_name navie_internlmxcomposer \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 50 \
    log_freq 10 \
    video_context datas/NExT-QA/nextqa/internvl_brief_val.json \
    is_video False \
    is_image True \
    vlm_type internlmxcomposer \
    question_type mc \
    mode ours_baseline