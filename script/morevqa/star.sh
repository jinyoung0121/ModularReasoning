CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node 1 --master_port=6245 morevqa.py \
    --options \
    exp_name debug \
    dataset.dataset_name STAR \
    dataset.data_path datas/STAR \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 10 \
    log_freq 1 \
    video_context datas/STAR/internlmxcomposer2_brief_allframe_val.json \
    image_vlm_type internlmxcomposer \
    mode morevqa