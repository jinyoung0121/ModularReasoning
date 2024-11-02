CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=8245 llm_only.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name TVQA \
    dataset.data_path datas/TVQA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 1 \
    internlm.max_batch_size 2 \
    video_context datas/TVQA/internlmxcomposer2_brief_val.json \
    llm_type internlm \
    mode llm_only

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=8245 jcef.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name TVQA \
    dataset.data_path datas/TVQA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    internlm.max_batch_size 1 \
    video_context datas/TVQA/internlmxcomposer2_brief_allframe_val.json \
    llm_type internlm \
    mode jcef