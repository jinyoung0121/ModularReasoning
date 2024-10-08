CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 morevqa_understanding.py  \
    --options \
    exp_name internlmxcomposer2_internlm_understanding_onlym1 \
    dataset.dataset_name ActivityNetQA \
    dataset.data_path datas/ActivityNet-QA \
    dataset.split test \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 4 \
    log_freq 1 \
    video_context datas/ActivityNet-QA/internlmxcomposer2_brief_val.json \
    is_video False \
    is_image True \
    vlm_type internlmxcomposer \
    question_type oe \
    retrieve_type univtg \
    mode morevqa_understanding