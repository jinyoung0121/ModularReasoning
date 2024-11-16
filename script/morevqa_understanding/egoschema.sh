CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.run --nproc_per_node 8 --master_port=9245 morevqa_understanding.py \
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
    video_context datas/EgoSchema/internlmxcomposer2_brief_val.json \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    retrieve_type univtg \
    llm_type internlm \
    internlm.max_batch_size 3 \
    frame_id_selection_num 16 \
    viclip.topk -1 \
    mode morevqa_understanding