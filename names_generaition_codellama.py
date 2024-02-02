from transformers import LlamaForCausalLM, CodeLlamaTokenizer, StoppingCriteria, StoppingCriteriaList
import pandas as pd
from tqdm import tqdm
import logging
import json
import warnings
import re
import torch
import time
import gc
import os
gc.collect()
df = pd.DataFrame(columns=['name_type', 'prompt', 'function_name', 'probs', 'ids', 'tokenised_name'])
import torch.nn.functional as F

LOG_FILE = '/home/sasha/effective-inference/clean_naming/logs/accurasies.log'
CODE_DATA_PATH = '/home/sasha/effective-inference/clean_naming/code_data.csv'
MODEL_NAME = "codellama/CodeLlama-7b-hf"
GENERATION_DATA_PREFIX = '/home/sasha/effective-inference/clean_naming/logs/generation_data_'

df = pd.DataFrame(columns=['name_type', 'prompt', 'function_name', 'probs', 'ids', 'tokenised_name'])
st = time.time()



# Logger setup
def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Model and Tokenizer setup
def setup_model_and_tokenizer():
    tokenizer = CodeLlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
    return model, tokenizer


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if len(scores) > 1 and self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False

def generate(prompt, model, tokenizer, stopping_criteria, max_new_tokens=30):
    # generation
    model.eval()
    with torch.no_grad():  # This ensures that no gradients are stored
        model.config.forced_eos_token_id = [13]
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
        # Generate ids
        generated_ids = model(input_ids)

        # Convert logits to probabilities
        probabilities = F.softmax(generated_ids['logits'], dim=-1)

        # breakpoint()
        # Extract probabilities for the generated tokens
        token_probabilities = [probabilities[0, i, e].detach().cpu().tolist() for i, e in enumerate(input_ids[0])]
        # Clear memory
        input_ids = input_ids.detach().cpu()
        torch.cuda.empty_cache()  # Clear CUDA cache

    ids = input_ids[0].tolist()

    return prompt, token_probabilities, ids
    # logging


def generation_step_next_token(i, p, name, code, type, model, tokenizer, stopping_criteria, fill_in_the_middle=False,  line=False):
    split_string = '\n' if line else (name + "(")
    # breakpoint()
    prompt = p + "\n" + split_string.join(code.split(split_string)[:i - 1]) + "<FILL_ME>"
    real = split_string + split_string.join(code.split(split_string)[i - 1:])
    if line : prompt+= "\n"
    if fill_in_the_middle and '\n' in real:
        prompt += '\n' + '\n'.join(real.split("\n")[1:])
    filling, scores, ids = generate(prompt, model, tokenizer, stopping_criteria)
    df.loc[len(df)] = {'name_type':type, 'prompt':prompt, 'function_name':name,  'probs': scores, 'ids': ids, 'tokenised_name': tokenizer(name, return_tensors="pt")["input_ids"].tolist()}


def process_row(ex, name, bad_name, numerical_name, translit_name, model, tokenizer, stopping_criteria, fill_in_the_middle=False, line=False):
    if line:
        lines = ex['code'].split('\n')
        matching_lines = [i for i, line in enumerate(lines) if re.search(fr"{name}\(", line)]
        n_steps = matching_lines
    else:
        n_steps = range(ex['prompt'].count(name + "("))

    for i in n_steps:
        # print(name, bad_name, numerical_name, "\n\n")
        generation_step_next_token(i, ex['prompt'], name, ex['code'], 'Original', model, tokenizer, stopping_criteria,
                                                           fill_in_the_middle, line)

        generation_step_next_token(i, ex['bad_prompt'], bad_name, ex['bad_code'],
                                                                'GPT generated', model, tokenizer, stopping_criteria, fill_in_the_middle, line)

        generation_step_next_token(i, ex['numerical_prompt'], numerical_name,
                                                            ex['numerical_code'], 'Numerical',model, tokenizer, stopping_criteria, fill_in_the_middle, line)

        generation_step_next_token(i, ex['translit_prompt'], translit_name, ex['translit_code'],
                                                           'Translit', model, tokenizer, stopping_criteria, fill_in_the_middle, line)
        # breakpoint()
        df.to_csv(f'/home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')

def main():
    setup_logging()
    model, tokenizer = setup_model_and_tokenizer()
    size = 0

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=torch.tensor([13]))])

    # Load data
    code_data = pd.read_csv(CODE_DATA_PATH, index_col=0)

    logging.info(f"---------------------------\n\n")
    logging.info(f"Logits collection\n")
    for j in tqdm(range(code_data.shape[0])):

        ex = code_data.loc[j]
        prompt_names_dict = json.loads(ex['prompt_names_dict'])
        prompt_numerical_dict = json.loads(ex['prompt_numerical_dict'])
        translit_names_dict = json.loads(ex['translit_names_dict'])

        for name, bad_name in prompt_names_dict.items():
            size+=1
            numerical_name = prompt_numerical_dict[name]
            translit_name = translit_names_dict[name]
            process_row(ex, name, bad_name, numerical_name, translit_name, model, tokenizer, stopping_criteria, fill_in_the_middle=False, line=True)



    logging.info(
        f'Generation info saved to file /home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
    logging.info(f"Dataset size is: {size}")
    print(f"Dataset size is: {size}")
    # Remember to close the file handler to release the resources
    logging.shutdown()


if __name__ == "__main__":
    main()