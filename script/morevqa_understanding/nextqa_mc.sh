# spatio-temporal retrieve (localize + UniVTG, prompt1: question input to UniVTG)
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 morevqa_understanding.py \
    --options \
    exp_name internlmxcomposer2_internlm_understanding_onlym1 \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
    is_video False \
    is_image True \
    vlm_type internlmxcomposer \
    retrieve_type univtg \
    mode morevqa_understanding