CUDA_VISIBLE_DEVICES=1 python llm_only.py \
    --options \
    exp_name internvl_internlm \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.batch_size 20 \
    dataset.start_sample 0 \
    internlm.max_batch_size 16 \
    log_freq 1 \
    video_context datas/internvl_brief.json \
    mode llm_only