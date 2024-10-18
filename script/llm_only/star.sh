CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port=6245 llm_only.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name STAR \
    dataset.data_path datas/STAR \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    internlm.max_batch_size 8 \
    video_context datas/STAR/internlmxcomposer2_brief_val.json \
    mode llm_only