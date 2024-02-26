from transformers import LlamaForCausalLM, CodeLlamaTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import logging
import json
import warnings
import re
import torch
import time
import outlines

LOG_FILE = '/home/sasha/effective-inference/clean_naming/logs/accurasies.log'
# CODE_DATA_PATH = '/home/sasha/effective-inference/clean_naming/functions_data.csv'
CODE_DATA_PATH = '/home/sasha/effective-inference/clean_naming/func_data_small.csv'
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
DEBUG = 0 # 0 or

df = pd.DataFrame(
        columns=['prompt', 'function_name', 'generated'])
st = time.time()


# Logger setup
def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Model and Tokenizer setup
def setup_model_and_tokenizer():
    tokenizer = ""#AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
    model = outlines.models.transformers(MODEL_NAME, model_kwargs = {'load_in_8bit':True})
    generator = outlines.generate.text(model)
    return generator



def generate(prompt, model, tokenizer, stopping_criteria, max_new_tokens=30):
    # generation

    input_ids = torch.tensor(tokenizer(prompt)["input_ids"]).unsqueeze(0).to('cuda')
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                   return_dict_in_generate=True, output_scores=True)
    filling = tokenizer.batch_decode(generated_ids.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return filling, [float(generated_ids['scores'][e][0][i].detach().cpu()) for e, i in
                     enumerate(generated_ids['sequences'][0][-len(generated_ids['scores']):])], \
        generated_ids['sequences'][0][-len(generated_ids['scores']):].tolist()


def generation_step_next_token(p, name, code, model, tokenizer, stopping_criteria, fill_in_the_middle=False,  line=False):
    split_string = (name + "(")
    prompt = '[INST]Rewrite this function.[/INST]\n'+ p.replace(split_string, "FILL_FUNCTION_NAME(")+'\n[ANS]'
    print(prompt)
    i = 0
    answer = model(prompt, max_tokens=20)

    while i<5 and 'def' not in answer :
        answer = model(prompt, max_tokens=20)
        i+=1
    
    print(answer)

    df.loc[len(df)] = {'prompt': prompt, 'function_name': name, 'generated': answer}



def process_row(ex, model, tokenizer, stopping_criteria, fill_in_the_middle=False, line=False):
    
    generation_step_next_token(ex['func_definition'],  ex['name'], '', model, tokenizer, stopping_criteria)
    df.to_csv(f'/home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')


def main():
    setup_logging()
    generator = setup_model_and_tokenizer()

    stopping_criteria = ''
    # Load data
    code_data = pd.read_csv(CODE_DATA_PATH, index_col=0).reset_index()

    logging.info(f"---------------------------\n\n")
    for j in tqdm(range(1002, code_data.shape[0])):
        ex = code_data.loc[j]
        process_row(ex, generator, '', stopping_criteria, fill_in_the_middle=True, line=True)
       
            
    logging.info(f"LLAMA instruct names generation")
    logging.info(
        f'Generation info saved to file /home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
    logging.info(f"Dataset size is: {df.shape}")

    # Remember to close the file handler to release the resources
    logging.shutdown()

if __name__ == "__main__":
    main()