# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port=9245 jcef.py \
#     --options \
#     exp_name internlmxcomposer2_internlm_16frames \
#     dataset.dataset_name NExTQA \
#     dataset.data_path datas/NExT-QA \
#     dataset.split val \
#     dataset.version multiplechoice \
#     dataset.fps 1 \
#     dataset.start_sample 0 \
#     batch_size 100 \
#     log_freq 10 \
#     internlm.max_batch_size 1 \
#     video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
#     mode jcef

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=9245 jdev.py \
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
#     internlm.max_batch_size 1 \
#     video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
#     llm_type internlm \
#     image_vlm_type internlmxcomposer \
#     video_vlm_type videollava \
#     mode jdev

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 jdev.py \
    --options \
    exp_name direct_questionanswer \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 200 \
    log_freq 20 \
    internlm.max_batch_size 1 \
    video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
    llm_type internlm \
    image_vlm_type internlmxcomposer \
    video_vlm_type videollava \
    mode jdev