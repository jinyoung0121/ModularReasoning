CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node 8 --master_port=4245 morevqa_understanding_CoT.py \
    --options \
    exp_name CoT \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 200 \
    log_freq 10 \
    video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
    video_context_vid datas/NExT-QA/nextqa/videollava_global_stepbystep_val.json \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    retrieve_type univtg \
    llm_type internlm \
    internlm.max_batch_size 3 \
    frame_id_selection_num 16 \
    viclip.topk -1 \
    mode morevqa_understanding