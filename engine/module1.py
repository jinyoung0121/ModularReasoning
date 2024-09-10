import math

def Module1(interpreter, module_input, external_memory):
    # program generation (API call z1 part)
    programs = interpreter.step_interpreters['llama'].generate(module_input, prompt_type='module1')

    # External Memory update using program execution output
    for program, memory in zip(programs, external_memory):
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(program, init_state=None)
            # update 'question' and 'frame_ids' field of External Memory when trim() != 'none'
            if output_state['TRIM0']['trim'] != '"none"':
                # update 'question' field
                memory['question'] = output_state['TRIM0']['truncated_question'][1:-1]
                
                # update 'frame_ids' field
                num_frames = int(len(memory['frame_ids'])*0.4) # in paper, mentioned 'truncating 40%'
                if output_state['TRIM0']['trim'] == '"beginning"':
                    start_idx = 0
                elif output_state['TRIM0']['trim'] == '"middle"':
                    start_idx = math.ceil(len(memory['frame_ids'])*0.3)
                elif output_state['TRIM0']['trim'] == '"end"':
                    start_idx = math.ceil(len(memory['frame_ids'])*0.6)
                else:
                    raise Exception('wrong trim option')
                memory['frame_ids'] = [i for i in range(start_idx, start_idx + num_frames)]
            
            # update 'conjunction' and 'event_queue' of External Memory when 'conj' != 'none'
            if output_state['PARSE_EVENT0']['conj'] != '"none"':
                # update 'conjunction' field of External Memory
                memory['conjunction'] = output_state['PARSE_EVENT0']['conj'][1:-1]
                
                # update ''question' and 'event_queue' field of External Memory
                memory['question'] = output_state['PARSE_EVENT0']['truncated_question'][1:-1]
                memory['event_queue'] = [output_state['PARSE_EVENT0']['event_to_localize'][1:-1], output_state['PARSE_EVENT0']['truncated_question'][1:-1]]
            else:
                # 'conj' == 'none'인 경우 일단 original question을 event_queue에 할당
                # TODO: question을 넣을 때 paraphrasing?
                memory['conjunction'] = output_state['PARSE_EVENT0']['conj'][1:-1]
                memory['event_queue'] = [memory['question']]
                
            # update 'require_ocr' field of External Memory
            if output_state['REQUIRE_OCR0'] != '"no"':
                memory['require_ocr'] = True
            
            # update 'qa_type' field of External Memory
            memory['qa_type'] = output_state['CLASSIFY0'][1:-1]
        except:
            memory['frame_ids'] = []
            memory['event_queue'] = [memory['question']]
            memory['qa_type'] = memory['question'].split(' ')[0][1:]
            memory['error'] = 'module1'
    
    return external_memory