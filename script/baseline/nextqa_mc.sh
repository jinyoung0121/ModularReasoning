CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run --nproc_per_node 1 --master_port=8245 baseline.py \
    --options \
    exp_name navie_videollava \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 10 \
    log_freq 1 \
    video_context datas/NExT-QA/nextqa/internvl_brief_val.json \
    is_video False \
    is_image True \
    vlm_type videollava \
    question_type mc \
    mode baseline