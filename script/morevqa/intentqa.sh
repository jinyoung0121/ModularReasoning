CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node 8 --master_port=1245 morevqa.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name IntentQA \
    dataset.data_path datas/IntentQA \
    dataset.split test \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/IntentQA/internlmxcomposer2_brief_test.json \
    image_vlm_type internlmxcomposer \
    llm_type internlm \
    mode morevqa