CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=6245 llm_only.py \
    --options \
    exp_name internvl_internlm \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.batch_size 100 \
    dataset.start_sample 0 \
    log_freq 10 \
    internlm.max_batch_size 8 \
    video_context datas/internvl_brief.json \
    mode llm_only