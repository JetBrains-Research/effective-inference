from transformers import LlamaForCausalLM, CodeLlamaTokenizer, StoppingCriteria, StoppingCriteriaList
import pandas as pd
from tqdm import tqdm
import logging
import json
import warnings
import re
import torch
import time

LOG_FILE = '/home/sasha/effective-inference/clean_naming/logs/accurasies.log'
CODE_DATA_PATH = '/home/sasha/effective-inference/clean_naming/code_data.csv'
MODEL_NAME = "codellama/CodeLlama-7b-hf"
GENERATION_DATA_PREFIX = '/home/sasha/effective-inference/clean_naming/logs/generation_data_'
DEBUG = 2 # 0 or n_examples for debugging


df = pd.DataFrame(
        columns=['name_type', 'prompt', 'function_name', 'real', 'generated', 'answer', 'scores', 'ids',
                 'tokenized_name'])
st = time.time()


# Logger setup
def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Model and Tokenizer setup
def setup_model_and_tokenizer():
    tokenizer = CodeLlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True)
    return model, tokenizer


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if len(scores) > 2 and self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False

def generate(prompt, model, tokenizer, stopping_criteria, max_new_tokens=30):
    # generation
    model.config.forced_eos_token_id = [13]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
    labels = torch.tensor([1]).unsqueeze(0)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, labels=labels,
                                   return_dict_in_generate=True, output_scores=True,
                                   stopping_criteria=stopping_criteria)

    filling = tokenizer.batch_decode(generated_ids.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    return filling, [float(generated_ids['scores'][e][0][i].detach().cpu()) for e, i in
                     enumerate(generated_ids['sequences'][0][-len(generated_ids['scores']):])], \
        generated_ids['sequences'][0][-len(generated_ids['scores']):].tolist()


def generation_step_next_token(i, p, name, code, type, model, tokenizer, stopping_criteria, fill_in_the_middle=False,  line=False):
    split_string = '\n' if line else (name + "(")
    prompt = p + "\n" + split_string.join(code.split(split_string)[:i - 1]) + "<FILL_ME>"
    real = code.split(split_string)[i] if line else (name)
    if line : prompt+= "\n"
    if fill_in_the_middle and '\n' in real:
        prompt += '\n' + '\n'.join(real.split("\n")[1:])
    filling, scores, ids = generate(prompt, model, tokenizer, stopping_criteria)
    if line:
        answer = (name+"(") in filling
    else:
        answer = filling[:len(name)] == name
    
    df.loc[len(df)] = {'name_type': type, 'prompt': prompt, 'function_name': name, 'real': real, 'generated': filling,
                       'answer': answer, 'scores': scores, 'ids': ids,
                       'tokenized_name': tokenizer(name, return_tensors="pt")["input_ids"].tolist()}
    return int(answer)


def process_row(ex, name, bad_name, numerical_name, translit_name, llama_name, model, tokenizer, stopping_criteria, acc_dict, fill_in_the_middle=False, line=False):
    if line:
        lines = ex['code'].split('\n')
        matching_lines = [i for i, line in enumerate(lines) if re.search(fr"{name}\(", line)]
        n_steps = matching_lines
    else:
        n_steps = range(ex['prompt'].count(name + "("))

    for i in n_steps:
        # print(name, bad_name, numerical_name, "\n\n")
        acc_dict['all'] += 1

        acc_dict['Original'] += generation_step_next_token(i, ex['prompt'], name, ex['code'], 'Original', model, tokenizer, stopping_criteria,
                                                           fill_in_the_middle, line)

        acc_dict['GPT generated'] += generation_step_next_token(i, ex['bad_prompt'], bad_name, ex['bad_code'],
                                                                'GPT generated', model, tokenizer, stopping_criteria, fill_in_the_middle, line)

        acc_dict['Numerical'] += generation_step_next_token(i, ex['numerical_prompt'], numerical_name,
                                                            ex['numerical_code'], 'Numerical',model, tokenizer, stopping_criteria, fill_in_the_middle, line)

        acc_dict['Translit'] += generation_step_next_token(i, ex['translit_prompt'], translit_name, ex['translit_code'],
                                                           'Translit', model, tokenizer, stopping_criteria, fill_in_the_middle, line)

        acc_dict['Llama'] += generation_step_next_token(i, ex['llama_prompt'], llama_name, ex['llama_code'],
                                                           'Llama', model, tokenizer, stopping_criteria, fill_in_the_middle, line)

        df.to_csv(f'/home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
        return acc_dict


def main():
    setup_logging()
    model, tokenizer = setup_model_and_tokenizer()

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=torch.tensor([13]))])

    # Load data
    code_data = pd.read_csv(CODE_DATA_PATH, index_col=0)
    n_examples = DEBUG if DEBUG else code_data.shape[0]
    acc_dict = {'all': 0, 'Llama':0, 'Original': 0, 'GPT generated': 0, 'Numerical': 0, 'Translit': 0}
    fill_in_the_middle=False
    line=True

    logging.info(f"---------------------------\n\n")
    logging.info(f'{"Fill in the middle" if fill_in_the_middle else "Code completion"}\n')
    logging.info(f'{"Next Line" if line else "Next Word"}\n')
    for i in ['prompt_names_dict', 'prompt_numerical_dict', 'translit_names_dict' ,'llama_names_dict' ]:
        code_data[i] = code_data[i].apply(json.loads)
    for j in tqdm(range(n_examples)):
        ex = code_data.loc[j]
        for name, bad_name in ex['prompt_names_dict'].items():
            numerical_name = ex['prompt_numerical_dict'][name]
            translit_name = ex['translit_names_dict'][name]
            llama_name = ex['llama_names_dict'][name]
            acc_dict = process_row(ex, name, bad_name, numerical_name, translit_name,llama_name,  model, tokenizer, stopping_criteria, acc_dict, fill_in_the_middle=fill_in_the_middle, line=line)

    logging.info(
        f'Generation info saved to file /home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
    logging.info(f"Dataset size is: {acc_dict['all']}")
    print(acc_dict)
    
    print(f"Original functions: {acc_dict['Original'] / acc_dict['all']}")
    logging.info(f"Original functions: {acc_dict['Original'] / acc_dict['all']}")

    print(f"GPT generated functions: {acc_dict['GPT generated'] / acc_dict['all']}")
    logging.info(f"GPT generated functions: {acc_dict['GPT generated'] / acc_dict['all']}")

    print(f"Numerical functions: {acc_dict['Numerical'] / acc_dict['all']}")
    logging.info(f"Numerical functions: {acc_dict['Numerical'] / acc_dict['all']}")

    print(f"Translit functions: {acc_dict['Translit'] / acc_dict['all']}")
    logging.info(f"Translit functions: {acc_dict['Translit'] / acc_dict['all']}")

    print(f"Llama functions: {acc_dict['Llama'] / acc_dict['all']}")
    logging.info(f"Llama functions: {acc_dict['Llama'] / acc_dict['all']}")

    # Remember to close the file handler to release the resources
    logging.shutdown()

if __name__ == "__main__":
    main()