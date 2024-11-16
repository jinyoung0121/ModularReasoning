from .step_interpreters import InternLM, Qwen, VideoLLaVA, load_model, unload_model
import util
import torch.distributed as dist

def Global_planning(config, EXTERNAL_MEMORY, **kwargs):
    # load model
    model = InternLM(config, device=kwargs['device'])
    model = load_model(model, kwargs['device'], config)
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[global planning generation]'
    # generate programs
    plannings = []
    stage_lists = []
    planning_lists = []
    for i, data in enumerate(metric_logger.log_every(dataloader, 1, header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        plannings = model.generate(data, prompt_type=kwargs['prompt_type'])
        planning_lists += plannings
        # only save stage name
        stage_lists += [util.set_stage(planning) for planning in plannings]
    
    # update process_stage: which stage to process
    for j, (plan, stage_list) in enumerate(zip(planning_lists, stage_lists)):
        try:
            EXTERNAL_MEMORY[j]['planning'] = plan
            for stage in stage_list:
                EXTERNAL_MEMORY[j][f'process_{stage.lower()}'] = True
        except:
            # when error occurs in the planning stage, set to only process final reasoning
            EXTERNAL_MEMORY[j][f'planning_error'] = True
            EXTERNAL_MEMORY[j][f'process_stage4'] = True
    
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)

    return EXTERNAL_MEMORY
    
def Program_generation(config, **kwargs):
    # load model
    if config.llm_type == 'internlm':
        model = InternLM(config, device=kwargs['device'])
        model = load_model(model, kwargs['device'], config)
    elif config.llm_type == 'qwen':
        model = Qwen(config, device=kwargs['device'])
    else:
        raise Exception('Invalid LLM type')
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[{} program generation]'.format(kwargs['prompt_type'])
    # generate programs
    programs = []
    for i, data in enumerate(metric_logger.log_every(dataloader, 1, header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # select valid inputs where the is_process==True
        valid_datas = [(j, d) for j, d in enumerate(data) if d['is_process']]
        # only pass valid inputs(understandings) to the model
        valid_program = [program for _, program in valid_datas]
        valid_outputs = model.generate(valid_program, prompt_type=kwargs['prompt_type'], num_options=config.dataset.num_options)
        results = ['none'] * len(data)
        for (k, _), output in zip(valid_datas, valid_outputs):
            results[k] = output
        programs += results
    
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return programs

def Program_generation_CoT(config, **kwargs):
    # load model
    if config.llm_type == 'internlm':
        model = InternLM(config, device=kwargs['device'])
        model = load_model(model, kwargs['device'], config)
    elif config.llm_type == 'qwen':
        model = Qwen(config, device=kwargs['device'])
    else:
        raise Exception('Invalid LLM type')
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[{} program generation]'.format(kwargs['prompt_type'])
    # generate programs
    programs = []
    for i, data in enumerate(metric_logger.log_every(dataloader, 1, header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # select valid inputs where the is_process==True
        valid_datas = [(j, d) for j, d in enumerate(data) if d['is_process']]
        # only pass valid inputs(understandings) to the model
        valid_program = [program for _, program in valid_datas]
        valid_outputs = model.generate(valid_program, prompt_type=kwargs['prompt_type'], num_options=config.dataset.num_options)
        results = ['none'] * len(data)
        for (k, _), output in zip(valid_datas, valid_outputs):
            if 'CoT' in kwargs['prompt_type']:
                try:
                    results[k] = output.split('The answer is:\n')[1]
                except:
                    results[k] = 'CODE ERROR!'
            else:
                results[k] = output
        programs += results
    
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return programs

def Understanding_generation(config, **kwargs):
    # load model
    if config.llm_type == 'internlm':
        model = InternLM(config, device=kwargs['device'])
        model = load_model(model, kwargs['device'], config)
    elif config.llm_type == 'qwen':
        model = Qwen(config, device=kwargs['device'])
    else:
        raise Exception('Invalid LLM type')
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    und_dataset = util.CustomDataset(kwargs['data'])
    und_dataloader = util.make_loader(und_dataset, config.batch_size // config.log_freq, config)
    
    und_metric_logger = util.MetricLogger(delimiter='  ')
    und_header = '[{} generation]'.format(kwargs['prompt_type']+'_understanding')
    # generate understanding
    understandings = []
    for i, data in enumerate(und_metric_logger.log_every(und_dataloader, 1, und_header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # select valid inputs where the is_process==True
        valid_datas = [(j, d) for j, d in enumerate(data) if d['is_process']]
        # only pass valid inputs(understandings) to the model
        valid_understand = [understand for _, understand in valid_datas]
        valid_outputs = model.generate(valid_understand, prompt_type=kwargs['prompt_type']+'_understanding', num_options=config.dataset.num_options)
        results = ['none'] * len(data)
        for (k, _), output in zip(valid_datas, valid_outputs):
            results[k] = output
        understandings += results
    
    und_metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()

    # prepare data for program generation
    questions = [d['question'] for d in kwargs['data']]
    qa_types = [d['qa_type'] for d in kwargs['data']]
    prog_inputs = [{'question': question, 'qa_type': qa_type, 'understanding': understanding} for question, qa_type, understanding in zip(questions, qa_types, understandings)]
    
    # make data iterable (bsz: batch_size//log_freq)
    prog_dataset = util.CustomDataset(prog_inputs)
    prog_dataloader = util.make_loader(prog_dataset, config.batch_size // config.log_freq, config)
    
    prog_metric_logger = util.MetricLogger(delimiter='  ')
    prog_header = '[{} program generation]'.format(kwargs['prompt_type'])
    # generate program
    programs = []
    for i, data in enumerate(prog_metric_logger.log_every(prog_dataloader, 1, prog_header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # select valid inputs where the input query(question) is not 'none'
        valid_datas = [(j, d) for j, d in enumerate(data) if d['understanding']!='none']
        # only pass valid inputs(programs) to the model
        valid_program = [program for _, program in valid_datas]
        valid_outputs = model.generate(valid_program, prompt_type=kwargs['prompt_type'], num_options=config.dataset.num_options)
        results = ['none'] * len(data)
        for (k, _), output in zip(valid_datas, valid_outputs):
            results[k] = output
        programs += results
    
    prog_metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return understandings, programs

def StageProgram_generation(config, **kwargs):
    # load model
    if config.llm_type == 'internlm':
        model = InternLM(config, device=kwargs['device'])
        model = load_model(model, kwargs['device'], config)
    elif config.llm_type == 'qwen':
        model = Qwen(config, device=kwargs['device'])
    else:
        raise Exception('Invalid LLM type')
    model.eval()
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[{} program generation]'.format(kwargs['prompt_type'])
    # generate program
    programs = []
    for i, data in enumerate(metric_logger.log_every(dataloader, 1, header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # select valid inputs where the input query(question) is not 'none'
        valid_datas = [(j, d) for j, d in enumerate(data) if d['is_process']]
        # only pass valid inputs(programs) to the model
        valid_program = [program for _, program in valid_datas]
        valid_outputs = model.generate(valid_program, prompt_type=kwargs['prompt_type'], num_options=config.dataset.num_options)
        results = ['none'] * len(data)
        for (k, _), output in zip(valid_datas, valid_outputs):
            results[k] = output.split('Answer:\n')[1]
        programs += results
    
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return programs

def VideoCaptioning(config, **kwargs):
    # load model
    if config.video_vlm_type == 'videollava':
        model = VideoLLaVA(config, device=kwargs['device'])
    else:
        raise Exception('Invalid model type')
    model = load_model(model, kwargs['device'], config)
    model.eval()
    
    # make data iterable (bsz: batch_size//log_freq)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, config.batch_size // config.log_freq, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[video captioning]'
    # generate captions
    captions = []
    for i, data in enumerate(metric_logger.log_every(dataloader, 1, header)):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # select valid inputs where the is_process==True
        valid_datas = [(j, d) for j, d in enumerate(data) if d['is_process']]
        # only pass valid inputs(understandings) to the model
        valid_inputs = [data for _, data in valid_datas]
        valid_outputs = model.video_captioning(valid_inputs)
        results = ['none'] * len(data)
        for (k, _), output in zip(valid_datas, valid_outputs):
            results[k] = output
        captions += results

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    unload_model(model)
    
    # return result
    return captions