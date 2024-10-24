# CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 morevqa.py \
#     --options \
#     exp_name debug \
#     dataset.dataset_name NExTQA \
#     dataset.data_path datas/NExT-QA \
#     dataset.split val \
#     dataset.version multiplechoice \
#     dataset.fps 1 \
#     dataset.start_sample 0 \
#     batch_size 40 \
#     log_freq 10 \
#     video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
#     image_vlm_type internlmxcomposer \
#     mode morevqa

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=9245 morevqa.py \
    --options \
    exp_name baseline_qwen \
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
    llm_type qwen \
    mode morevqa