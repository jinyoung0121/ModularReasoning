CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=7245 morevqa_understanding_saved.py \
    --options \
    exp_name debugdebug_noimgcontext \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 200 \
    log_freq 20 \
    video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    retrieve_type univtg \
    llm_type internlm \
    frame_id_selection_num 16 \
    mode morevqa_understanding