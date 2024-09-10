

def Module3(interpreter, module_input, external_memory, batch):
    # program generation (API call z3 part)
    programs = interpreter.step_interpreters['llama'].generate(module_input, prompt_type='module3')
    VLM_answers = []
    # External Memory update using program execution output
    for program, memory, visual in zip(programs, external_memory, batch['image']):
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(program, init_state={'image': visual, 'frame_ids': memory['frame_ids']})
            # heuristic하게 작성. only two output exist. VQA0, VQA1
            QA_pools = []
            QA_pools += output_state['VQA0']
            QA_pools += output_state['VQA1']
            
            # sort in ascending order based on frame_id
            sorted_QA_pools = sorted(QA_pools, key=lambda x:x['frame_id'])
            
            # answer formatting
            answers = []
            for qa in sorted_QA_pools:
                answers.append(f"[frame{qa['frame_id']:>4}]{qa['question']}: {qa['answer']}")
        except:
            memory['frame_ids'] = []
            memory['error'] = 'module3'
            answers = ['']
        VLM_answers.append('\n'.join(answers))
    
    return external_memory, VLM_answers