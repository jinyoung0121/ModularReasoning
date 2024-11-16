CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=1245 morevqa_understanding.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name IntentQA \
    dataset.data_path datas/IntentQA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 200 \
    log_freq 20 \
    video_context datas/IntentQA/internlmxcomposer2_brief_val.json \
    video_context_vid datas/IntentQ/videollava_global_stepbystep_val.json \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    retrieve_type univtg \
    llm_type internlm \
    internlm.max_batch_size 3 \
    frame_id_selection_num 16 \
    viclip.topk -1 \
    mode morevqa_understanding