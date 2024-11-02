CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 --master_port=6245 llm_only.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name IntentQA \
    dataset.data_path datas/IntentQA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 10 \
    log_freq 1 \
    internlm.max_batch_size 3 \
    video_context datas/STAR/internlmxcomposer2_brief_val.json \
    llm_type internlm \
    mode llm_only