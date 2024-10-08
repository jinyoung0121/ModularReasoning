from .step_interpreters import InternLM, InternLM2, load_model, unload_model
import util
import torch.distributed as dist

def Program_generation(config, **kwargs):
    # load model
    if config.mode in ['jcef', 'morevqa', 'morevqa_retrieve']:
        model = InternLM(config, device=kwargs['device'])
    elif config.mode in ['morevqa_understanding']:
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

def Understanding_generation(config, **kwargs):
    # load model
    if config.mode in ['jcef', 'morevqa', 'morevqa_retrieve']:
        model = InternLM(config, device=kwargs['device'])
    elif config.mode in ['morevqa_understanding']:
        model = InternLM2(config, device=kwargs['device'])
    model = load_model(model, kwargs['device'], config)
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    und_dataset = util.CustomDataset(kwargs['data'])
    und_dataloader = util.make_loader(und_dataset, config.batch_size // config.log_freq, config)
    
    und_metric_logger = util.MetricLogger(delimiter='  ')
    und_header = '[{} generation]'.format(kwargs['prompt_type']+'_understanding')
    # generate understanding
    understandings = []
    for i, understanding in enumerate(und_metric_logger.log_every(und_dataloader, 1, und_header)):
        understandings += model.generate(understanding, prompt_type=kwargs['prompt_type']+'_understanding')
    
    und_metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    
    # prepare data for program generation
    questions = [d['question'] for d in kwargs['data']]
    prog_inputs = [{'question': question, 'understanding': understanding} for question, understanding in zip(questions, understandings)]
    
    # make data iterable (bsz: batch_size//log_freq)
    prog_dataset = util.CustomDataset(prog_inputs)
    prog_dataloader = util.make_loader(prog_dataset, config.batch_size // config.log_freq, config)
    
    prog_metric_logger = util.MetricLogger(delimiter='  ')
    prog_header = '[{} program generation]'.format(kwargs['prompt_type'])
    # generate program
    programs = []
    for i, program in enumerate(prog_metric_logger.log_every(prog_dataloader, 1, und_header)):
        programs += model.generate(program, prompt_type=kwargs['prompt_type'])
    
    prog_metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    
    # return result
    return understandings, programs