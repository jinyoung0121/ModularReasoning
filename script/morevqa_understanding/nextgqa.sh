CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 morevqa_understanding.py \
    --options \
    exp_name internlmxcomposer2_internlm_understanding_onlym1 \
    dataset.dataset_name NExTGQA \
    dataset.data_path datas/NExT-GQA \
    dataset.video_path datas/NExT-QA \
    dataset.split test \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    dataset.max_samples 10 \
    eval_grounding True \
    batch_size 10 \
    log_freq 1 \
    video_context datas/internlmxcomposer2_brief_val.json \
    is_video False \
    is_image True \
    vlm_type internvl \
    vlm_type internlmxcomposer \
    retrieve_type univtg \
    mode morevqa_understanding