
def load_baseline_llm_prompt(data, prompt_type, config, num_options=5):
    if prompt_type == 'module1':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/module1_baseline.prompt'
    elif prompt_type == 'module2':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/module2_baseline.prompt'
    elif prompt_type == 'module2_retrieve':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/module2_spatioretrieve.prompt'
    elif prompt_type == 'module3':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/module3_baseline.prompt'
    elif prompt_type == 'final':
        additional_system_prompt = 'Only answer with the final answer.'
        if config.question_type == 'mc':
            prompt_path = f'datas/prompt/final_prediction_mc_{str(num_options)}.prompt'
        elif config.question_type == 'oe':
            prompt_path = 'datas/prompt/final_prediction_oe.prompt'
        else:
            raise Exception('Invalid question type!')
    elif prompt_type == 'llm_only':
        additional_system_prompt = 'Only answer with the final answer.'
        if config.question_type == 'mc':
            prompt_path = f'datas/prompt/llm_only_mc_{str(num_options)}.prompt'
        elif config.question_type == 'oe':
            prompt_path = 'datas/prompt/llm_only_oe.prompt'
        else:
            raise Exception('Invalid question type!')
    else:
        raise Exception('wrong prompt type')
    with open(prompt_path) as f:
        base_prompt = f.read().strip()

    if prompt_type == 'module1':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module2':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                .replace('INSERT_PHRASE1_HERE', d['event_queue'][0])
                                .replace('INSERT_PHRASE2_HERE', d['event_queue'][1]) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module2_retrieve':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                .replace('INSERT_EVENT1_HERE', d['event_queue'][0])
                                .replace('INSERT_EVENT2_HERE', d['event_queue'][1]) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module3':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'final':
        if isinstance(data, list):
            if config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']).replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
            elif config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception('Invalid question type!')
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'llm_only':
        if isinstance(data, list):
            if config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
            elif config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception('Invalid question type!')
        else:
            raise TypeError('data must be list of strings')
    else:   
        raise Exception('wrong prompt type')
    
    return prompt, additional_system_prompt

def load_llm_prompt(data, prompt_type, config, num_options=5):
    if prompt_type == 'planning':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/planning_global.prompt'
    elif prompt_type == 'stage1_nounderstanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage1_nounderstanding.prompt'
    elif prompt_type == 'stage1_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage1_understanding.prompt'
    elif prompt_type == 'stage1':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage1.prompt'
    elif prompt_type == 'stage1_CoT':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage1_CoT.prompt'
    elif prompt_type == 'stage2_nounderstanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage2_nounderstanding.prompt'
    elif prompt_type == 'stage2_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage2_understanding.prompt'
    elif prompt_type == 'stage2':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage2.prompt'
    elif prompt_type == 'stage2_CoT':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage2_CoT.prompt'
    elif prompt_type == 'stage3_nounderstanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage3_nounderstanding.prompt'
    elif prompt_type == 'stage3_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage3_understanding.prompt'
    elif prompt_type == 'stage3':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage3.prompt'
    elif prompt_type == 'stage3_CoT':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage3_CoT.prompt'
    elif prompt_type == 'stage4_nounderstanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage4_nounderstanding.prompt'
    elif prompt_type == 'stage4_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage4_understanding.prompt'
    elif prompt_type == 'stage4':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage4.prompt'
    elif prompt_type == 'stage4_CoT':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage4_CoT.prompt'
    elif prompt_type == 'final':
        additional_system_prompt = 'Only answer with the final answer.'
        if config.question_type == 'mc':
            prompt_path = f'datas/prompt/final_prediction_mc_{str(num_options)}.prompt'
        elif config.question_type == 'oe':
            prompt_path = 'datas/prompt/final_prediction_oe.prompt'
        else:
            raise Exception('Invalid question type!')
    elif prompt_type == 'llm_only':
        additional_system_prompt = 'Only answer with the final answer.'
        if config.question_type == 'mc':
            prompt_path = f'datas/prompt/llm_only_mc_{str(num_options)}.prompt'
        elif config.question_type == 'oe':
            prompt_path = 'datas/prompt/llm_only_oe.prompt'
        else:
            raise Exception('Invalid question type!')
    else:
        raise Exception('wrong prompt type')
    with open(prompt_path) as f:
        base_prompt = f.read().strip()
    
    if prompt_type in ['planning', 'stage1_understanding', 'stage2_understanding', 'stage3_understanding', 'stage1_CoT', 'stage2_CoT', 'stage3_CoT', 'stage1_nounderstanding', 'stage2_nounderstanding', 'stage3_nounderstanding']:
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type in ['stage4_understanding', 'stage4_CoT', 'stage4_nounderstanding']:
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type in ['stage1', 'stage2', 'stage3']:
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).replace('INSERT_UNDERSTANDING_HERE', d['understanding']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type in ['stage4', 'stage4_image', 'stage4_video']:
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']).replace('INSERT_UNDERSTANDING_HERE', d['understanding']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module2':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                .replace('INSERT_EVENT1_HERE', d['event_queue'][0])
                                .replace('INSERT_EVENT2_HERE', d['event_queue'][1]) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module3':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'final':
        if isinstance(data, list):
            if config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context'])
                                    .replace('INSERT_QUESTION_HERE', d['question'])
                                    .format(len_options=len(d['option']), options=d['option']) for d in data]
            elif config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context'])
                                    .replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception('Invalid question type!')
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'llm_only':
        if isinstance(data, list):
            if config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question'])
                                    .format(len_options=len(d['option']), options=d['option']) for d in data]
            elif config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception("Invalid question type!")
        else:
            raise TypeError('data must be list of strings')
    else:   
        raise Exception('wrong prompt type')
    
    return prompt, additional_system_prompt

def load_llm_prompt_qwen(data, prompt_type, config, num_options=5):
    if prompt_type == 'planning':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/planning_global.prompt'
    elif prompt_type == 'stage1_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage1_understanding_qwen.prompt'
    elif prompt_type == 'stage1':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage1_qwen.prompt'
    elif prompt_type == 'stage2_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage2_understanding_qwen.prompt'
    elif prompt_type == 'stage2':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage2_qwen.prompt'
    elif prompt_type == 'stage3_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage3_understanding_qwen.prompt'
    elif prompt_type == 'stage3':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage3_qwen.prompt'
    elif prompt_type == 'stage4_understanding':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage4_understanding.prompt'
    elif prompt_type == 'stage4':
        additional_system_prompt = 'Only answer with the final answer similar to given examples.'
        prompt_path = 'datas/prompt/stage4.prompt'
    elif prompt_type == 'final':
        additional_system_prompt = 'Only answer with the final answer.'
        if config.question_type == 'mc':
            prompt_path = f'datas/prompt/final_prediction_mc_{str(num_options)}_qwen{config.qwen.model_size}.prompt'
        elif config.question_type == 'oe':
            prompt_path = 'datas/prompt/final_prediction_oe.prompt'
        else:
            raise Exception('Invalid question type!')
    elif prompt_type == 'llm_only':
        additional_system_prompt = 'Only answer with the final answer.'
        if config.question_type == 'mc':
            prompt_path = f'datas/prompt/llm_only_mc_{str(num_options)}_qwen.prompt'
        elif config.question_type == 'oe':
            prompt_path = 'datas/prompt/llm_only_oe.prompt'
        else:
            raise Exception('Invalid question type!')
    else:
        raise Exception('wrong prompt type')
    with open(prompt_path) as f:
        base_prompt = f.read().strip()
    
    if prompt_type in ['planning', 'stage1_understanding', 'stage2_understanding', 'stage3_understanding']:
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'stage4_understanding':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type in ['stage1', 'stage2', 'stage3', 'stage4']:
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).replace('INSERT_UNDERSTANDING_HERE', d['understanding']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'stage4':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']).replace('INSERT_UNDERSTANDING_HERE', d['understanding']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module2':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                .replace('INSERT_EVENT1_HERE', d['event_queue'][0])
                                .replace('INSERT_EVENT2_HERE', d['event_queue'][1]) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'module3':
        if isinstance(data, list):
            prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'final':
        if isinstance(data, list):
            if config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context'])
                                    .replace('INSERT_QUESTION_HERE', d['question'])
                                    .format(len_options=len(d['option']), options=d['option']) for d in data]
            elif config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context'])
                                    .replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception('Invalid question type!')
        else:
            raise TypeError('data must be list of strings')
    elif prompt_type == 'llm_only':
        if isinstance(data, list):
            if config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question'])
                                    .format(len_options=len(d['option']), options=d['option']) for d in data]
            elif config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception("Invalid question type!")
        else:
            raise TypeError('data must be list of strings')
    else:   
        raise Exception('wrong prompt type')
    
    return prompt, additional_system_prompt