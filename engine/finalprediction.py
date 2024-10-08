from .step_interpreters import InternVLInterpreter, InternLMXComposerInterpreter, InternVideo, load_model, unload_model
import util
import torch.distributed as dist

def FinalPrediction(config, **kwargs):
    vlm_mapping = {'internvl': InternVLInterpreter, 'internlmxcomposer': InternLMXComposerInterpreter, 'internvideo': InternVideo}
    # load model
    model = vlm_mapping[config.vlm_type](config, kwargs['device'])
    model = load_model(model, kwargs['device'], config)
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Final prediction]'
    # generate answer
    answers = []
    for i, data in enumerate(metric_logger.log_every(dataloader, 1, header)):
        answers += model.video_predict(data)
        
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return answers