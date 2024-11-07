CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=9245 jcef.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name EgoSchema \
    dataset.data_path datas/EgoSchema \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 50 \
    log_freq 10 \
    internlm.max_batch_size 1 \
    video_context datas/EgoSchema/internlmxcomposer2_brief_allframe_val.json \
    llm_type internlm \
    mode jcef