CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port=6245 llm_only.py \
    --options \
    exp_name navie_videollava \
    dataset.dataset_name STAR \
    dataset.data_path datas/STAR \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/NExT-QA/nextqa/internvl_brief_val.json \
    is_video False \
    is_image True \
    vlm_type videollava \
    question_type mc \
    mode baseline