

def FinalPrediction(interpreter, module_input):
    # answer generation
    final_answers = interpreter.step_interpreters['llama'].generate(module_input, prompt_type='final')
    return final_answers