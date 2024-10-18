# # temporal retrieve (only UniVTG, prompt2: question -> phrase convert)
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node 6 --master_port=9245 morevqa_retrieve.py \
#     --options \
#     exp_name internlmxcomposer2_internlm_prompt2 \
#     dataset.dataset_name NExTQA \
#     dataset.data_path datas/NExT-QA \
#     dataset.split val \
#     dataset.version multiplechoice \
#     dataset.fps 1 \
#     dataset.start_sample 0 \
#     batch_size 100 \
#     log_freq 10 \
#     video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
#     is_video False \
#     is_image True \
#     vlm_type internlmxcomposer \
#     retrieve_type univtg \
#     mode morevqa_retrieve

# spatio-temporal retrieve (localize + UniVTG, prompt1: question input to UniVTG)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node 6 --master_port=9245 morevqa_retrieve.py \
    --options \
    exp_name internlmxcomposer2_internlm_spatio \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/NExT-QA/nextqa/internlmxcomposer2_brief_val.json \
    image_vlm_type internlmxcomposer \
    retrieve_type univtg \
    mode morevqa_retrieve