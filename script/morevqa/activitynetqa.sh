CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port=7245 morevqa.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name ActivityNetQA \
    dataset.data_path datas/ActivityNet-QA \
    dataset.split test \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 4 \
    log_freq 1 \
    video_context datas/ActivityNet-QA/internlmxcomposer2_brief_test.json \
    is_video False \
    is_image True \
    vlm_type internlmxcomposer \
    question_type oe \
    mode morevqa