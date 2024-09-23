import math
import torch
import torch.distributed as dist
from .utils import ProgramInterpreter
from .step_interpreters import unload_model
import util

def Stage1(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage1 processing]'

    # External Memory update using program execution output (iterate over only batch 1)
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(data['program'][0], init_state=None) # assume only batch 1
            # update 'question' and 'frame_ids' field of External Memory when trim() != 'none'
            if output_state['TRIM0']['trim'] != 'none':
                EXTERNAL_MEMORY[i]['question'] = output_state['TRIM0']['truncated_question']
            
                # update 'frame_ids' field
                num_frames = int(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.4) # like MoReVQA paper, mentioned 'truncating 40%'
                if output_state['TRIM0']['trim'] == 'beginning':
                    start_idx = 0
                elif output_state['TRIM0']['trim'] == 'middle':
                    start_idx = math.ceil(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.3)
                elif output_state['TRIM0']['trim'] == 'end':
                    start_idx = math.ceil(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.6)
                else:
                    raise Exception('wrong trim option')
                EXTERNAL_MEMORY[i]['frame_ids'] = [i for i in range(start_idx, start_idx + num_frames)]
            

            # update 'conjunction' field of External Memory
            EXTERNAL_MEMORY[i]['conjunction'] = output_state['PARSE_EVENT0']['conj']
                
            # update 'question' field
            EXTERNAL_MEMORY[i]['question'] = output_state['PARSE_EVENT0']['main_phrase']
                
            # update 'phrases' field
            anchor_phrase = output_state['PARSE_EVENT0']['anchor_phrase']
            main_phrase = output_state['PARSE_EVENT0']['main_phrase']
            EXTERNAL_MEMORY[i]['phrases'] = [anchor_phrase, main_phrase]
            
            # update 'require_ocr' field of External Memory
            if output_state['REQUIRE_OCR0'] != 'no':
                EXTERNAL_MEMORY[i]['require_ocr'] = True

            # update 'qa_type' field of External Memory
            EXTERNAL_MEMORY[i]['qa_type'] = output_state['CLASSIFY0']
        except:
            EXTERNAL_MEMORY[i]['frame_ids'] = []
            EXTERNAL_MEMORY[i]['phrases'] = ["", EXTERNAL_MEMORY[i]['question']]
            EXTERNAL_MEMORY[i]['qa_type'] = EXTERNAL_MEMORY[i]['question'].split(' ')[0][1:]
            EXTERNAL_MEMORY[i]['error'] = 'stage1'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)
    
    return EXTERNAL_MEMORY

def Stage2(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage2 processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # initialize frame_id (indicator)
        indicator = torch.zeros(data['image'][0].size(0))
        indicator[EXTERNAL_MEMORY[i]['frame_ids']] = 1
        
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(data['program'][0], init_state={'video_path': data['video_path'][0],
                                                                                             'indicator': indicator.bool()},) # assume only batch 1
            # update 'frame_ids' field
            EXTERNAL_MEMORY[i]['frame_ids'] = output_state['RETRIEVE1']
        except:
            EXTERNAL_MEMORY[i]['frame_ids'] = []
            if EXTERNAL_MEMORY[i]['error'] == None:
                EXTERNAL_MEMORY[i]['error'] = 'stage2'
                
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY

def Stage3(interpreter, module_input, external_memory, batch):
    
    return None, None

def Stage4(interpreter, module_input, external_memory, batch, config):
    # program generation (API call for stage4)
    programs = interpreter.step_interpreters['internlm'].generate(module_input, prompt_type='stage4')
    VLM_answers = {'video': [], 'image': []}
    # External Memory update using program execution output
    for program, memory, visual, video_path in zip(programs, external_memory, batch['image'], batch['video_path']):
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(program, init_state={'image': visual, 'frame_ids': memory['frame_ids'], 'is_video': config.use_video, 'is_image': config.use_image})
            # heuristic하게 작성. only two output exist. VQA0, VQA1
            QA_pools = {'video': [], 'image': []}
            QA_pools['video'] += output_state['VQA0']['video']
            QA_pools['video'] += output_state['VQA1']['video']
            QA_pools['image'] += output_state['VQA0']['image']
            QA_pools['image'] += output_state['VQA1']['image']
            
            # sort in ascending order based on frame_id
            sorted_QA_pools = sorted(QA_pools['image'], key=lambda x:x['frame_id'])
                
            # answer formatting (video)
            for qa in QA_pools['video']:
                VLM_answers['video'].append(f"[video]{qa['question']}: {qa['answer']}")
            # answer formatting (image)
            for qa in sorted_QA_pools:
                VLM_answers['image'].append(f"[frame{qa['frame_id']:>4}]{qa['question']}: {qa['answer']}")
        except:
            memory['frame_ids'] = []
            memory['error'] = 'stage4'
            
    return external_memory, VLM_answers