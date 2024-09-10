CUDA_VISIBLE_DEVICES=0 python jcef.py \
    --options \
    exp_name blip_llama \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.batch_size 1 \
    dataset.start_sample 0 \
    dataset.max_samples 20 \
    log_every 10 \
    video_context datas/blip2t5xxl_allframe.json \
    mode jcef