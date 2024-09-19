import math
import torch
import torch.distributed as dist
from .utils import ProgramInterpreter
from .step_interpreters import unload_model
import util

def Module1(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Module1 processing]'
    # External Memory update using program execution output (iterate over only batch 1)
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(data['program'][0], init_state=None) # assume only batch 1
            # update 'question' and 'frame_ids' field of External Memory when trim() != 'none'
            if output_state['TRIM0']['trim'] != 'none':
                # update 'question' field
                EXTERNAL_MEMORY[i]['question'] = output_state['TRIM0']['truncated_question'].split('')
                
                # update 'frame_ids' field
                num_frames = int(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.4) # in paper, mentioned 'truncating 40%'
                if output_state['TRIM0']['trim'] == 'beginning':
                    start_idx = 0
                elif output_state['TRIM0']['trim'] == 'middle':
                    start_idx = math.ceil(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.3)
                elif output_state['TRIM0']['trim'] == 'end':
                    start_idx = math.ceil(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.6)
                else:
                    raise Exception('wrong trim option')
                EXTERNAL_MEMORY[i]['frame_ids'] = [i for i in range(start_idx, start_idx + num_frames)]
            
            # update 'conjunction' and 'event_queue' of External Memory when 'conj' != 'none'
            if output_state['PARSE_EVENT0']['conj'] != 'none':
                # update 'conjunction' field of External Memory
                EXTERNAL_MEMORY[i]['conjunction'] = output_state['PARSE_EVENT0']['conj']
                
                # update ''question' and 'event_queue' field of External Memory
                EXTERNAL_MEMORY[i]['question'] = output_state['PARSE_EVENT0']['truncated_question']
                EXTERNAL_MEMORY[i]['event_queue'] = [output_state['PARSE_EVENT0']['event_to_localize'], output_state['PARSE_EVENT0']['truncated_question']]
            else:
                # 'conj' == 'none'인 경우 일단 original question을 event_queue에 할당
                # TODO: question을 넣을 때 paraphrasing?
                EXTERNAL_MEMORY[i]['conjunction'] = output_state['PARSE_EVENT0']['conj']
                EXTERNAL_MEMORY[i]['event_queue'] = ['none', EXTERNAL_MEMORY[i]['question']]
                
            # update 'require_ocr' field of External Memory
            if output_state['REQUIRE_OCR0'] != 'no':
                EXTERNAL_MEMORY[i]['require_ocr'] = True
            
            # update 'qa_type' field of External Memory
            EXTERNAL_MEMORY[i]['qa_type'] = output_state['CLASSIFY0']
        except:
            EXTERNAL_MEMORY[i]['frame_ids'] = []
            EXTERNAL_MEMORY[i]['event_queue'] = ['none', EXTERNAL_MEMORY[i]['question']]
            EXTERNAL_MEMORY[i]['qa_type'] = EXTERNAL_MEMORY[i]['question'].split(' ')[0][1:]
            EXTERNAL_MEMORY[i]['error'] = 'module1'
    
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)
    
    return EXTERNAL_MEMORY

def Module2(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Module2 processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # initalize frame_id (indicator)
        frames = data['image'][0]
        indicator = torch.zeros(frames.size(0))
        indicator[EXTERNAL_MEMORY[i]['frame_ids']] = 1
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(data['program'][0], init_state={'image': frames,
                                                                                             'indicator': indicator.bool(),}) # assume only batch 1
            # update 'frame_ids' field
            EXTERNAL_MEMORY[i]['frame_ids'] = output_state['VERIFY_ACTION0']
        except:
            EXTERNAL_MEMORY[i]['frame_ids'] = []
            if EXTERNAL_MEMORY[i]['error'] == None:
                EXTERNAL_MEMORY[i]['error'] = 'module2'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY

def Module3(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, device=kwargs['device'])
    
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Module3 processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(data['program'][0], init_state={'image': data['image'][0],
                                                                                             'frame_ids': EXTERNAL_MEMORY[i]['frame_ids'],
                                                                                             'is_video': config.is_video,
                                                                                             'is_image': config.is_image,}) # assume only batch 1
            # heuristic하게 작성. only two output exist. VQA0, VQA1
            # only using image VQA output in MoReVQA
            QA_pools = []
            QA_pools += output_state['VQA0']['image']
            QA_pools += output_state['VQA1']['image']
            
            # sort in ascending order based on frame_id
            sorted_QA_pools = sorted(QA_pools, key=lambda x:x['frame_id'])
            
            # answer formatting
            answers = []
            for qa in sorted_QA_pools:
                answers.append(f"[frame{qa['frame_id']:>4}]{qa['question']}: {qa['answer']}")
            EXTERNAL_MEMORY[i]['VLM_answer'] = '\n'.join(answers)
        except:
            EXTERNAL_MEMORY[i]['frame_ids'] = []
            if EXTERNAL_MEMORY[i]['error'] == None:
                EXTERNAL_MEMORY[i]['error'] = 'module3'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY