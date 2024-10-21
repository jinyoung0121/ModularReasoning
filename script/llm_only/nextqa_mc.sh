# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=6245 llm_only.py \
#     --options \
#     exp_name internlmxcomposer2_internlm \
#     dataset.dataset_name NExTQA \
#     dataset.data_path datas/NExT-QA \
#     dataset.split val \
#     dataset.version multiplechoice \
#     dataset.fps 1 \
#     dataset.start_sample 0 \
#     batch_size 100 \
#     log_freq 10 \
#     internlm.max_batch_size 8 \
#     video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
#     mode llm_only

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node 1 --master_port=6245 llm_only.py \
    --options \
    exp_name debug \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 10 \
    log_freq 1 \
    qwen.max_batch_size 1 \
    video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
    llm_type qwen \
    mode llm_only