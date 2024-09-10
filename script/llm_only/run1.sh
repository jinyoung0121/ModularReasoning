CUDA_VISIBLE_DEVICES=0 python llm_only.py \
    --options \
    exp_name blip_llama \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.batch_size 1 \
    dataset.start_sample 0 \
    log_every 10 \
    mode llm_only