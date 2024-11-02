CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --master_port=9245 jdev.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name IntentQA \
    dataset.data_path datas/IntentQA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    internlm.max_batch_size 1 \
    video_context datas/IntentQA/internlmxcomposer2_brief_val.json \
    llm_type internlm \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    mode jdev