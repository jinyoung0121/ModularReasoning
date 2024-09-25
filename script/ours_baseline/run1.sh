# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=5245 ours_baseline1.py \
#     --options \
#     exp_name internvideo_baseline1 \
#     dataset.dataset_name NExTQA \
#     dataset.data_path datas/NExT-QA \
#     dataset.split val \
#     dataset.version multiplechoice \
#     dataset.fps 1 \
#     dataset.start_sample 0 \
#     batch_size 100 \
#     log_freq 10 \
#     video_context datas/internvl_brief.json \
#     vlm_type internvideo \
#     mode ours_baseline

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=7245 ours_baseline2.py \
#     --options \
#     exp_name internvideo_baseline2 \
#     dataset.dataset_name NExTQA \
#     dataset.data_path datas/NExT-QA \
#     dataset.split val \
#     dataset.version multiplechoice \
#     dataset.fps 1 \
#     dataset.start_sample 0 \
#     batch_size 100 \
#     log_freq 10 \
#     video_context datas/internvl_brief.json \
#     vlm_type internvideo \
#     mode ours_baseline

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node 2 --master_port=9272 ours_baseline2.py \
    --options \
    exp_name internvl_baseline2 \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val_temporal \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/internvl_brief.json \
    vlm_type internvl \
    mode ours_baseline