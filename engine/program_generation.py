from .step_interpreters import InternLM, InternLM2, load_model, unload_model
import util
import torch.distributed as dist

def Program_generation(config, **kwargs):
    # load model
    if config.mode in ['jcef', 'morevqa']:
        model = InternLM(config, device=kwargs['device'])
    elif config.mode in ['ours_baseline', 'ours']:
        model = InternLM2(config, device=kwargs['device'])
    model = load_model(model, kwargs['device'], config)
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[{} program generation]'.format(kwargs['prompt_type'])
    # generate programs
    programs = []
    for i, program in enumerate(metric_logger.log_every(dataloader, 1, header)):
        programs += model.generate(program, prompt_type=kwargs['prompt_type'])
    
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return programs
