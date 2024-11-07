CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=7245 jdev.py \
    --options \
    exp_name internlmxcomposer2_internlm_fps05_min16 \
    dataset.dataset_name STAR \
    dataset.data_path datas/STAR \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    internlm.max_batch_size 1 \
    video_context datas/STAR/internlmxcomposer2_brief_halfframe_val.json \
    llm_type internlm \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    mode jdev