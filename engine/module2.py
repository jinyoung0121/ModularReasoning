import torch

def Module2(interpreter, module_input, external_memory, batch):
    # program generation (API call z2 part)
    programs = interpreter.step_interpreters['llama'].generate(module_input, prompt_type='module2')
    # External Memory update using program execution output
    for program, memory, visual in zip(programs, external_memory, batch['image']):
        # initalize frame_id (indicator)
        indicator = torch.zeros(visual.size(0))
        indicator[memory['frame_ids']] = 1
        try:
            # get output by program execution
            final_output, output_state = interpreter.execute(program, init_state={'image': visual, 'indicator': indicator.bool()})
            # update 'frame_ids' field
            memory['frame_ids'] = output_state['VERIFY_ACTION0']
        except:
            memory['frame_ids'] = []
            memory['error'] = 'module2'
    
    return external_memory