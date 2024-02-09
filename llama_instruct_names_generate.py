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
CODE_DATA_PATH = '/home/sasha/effective-inference/clean_naming/functions_data.csv'
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
DEBUG = 0 # 0 or

df = pd.DataFrame(
        columns=['prompt', 'function_name', 'generated', 'scores', 'ids'])
st = time.time()


# Logger setup
def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Model and Tokenizer setup
def setup_model_and_tokenizer():
    tokenizer = ""#AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
    model = outlines.models.transformers(MODEL_NAME, device={"":0}, model_kwargs = {'load_in_8bit':True})
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

    input_ids = torch.tensor(tokenizer(prompt)["input_ids"]).unsqueeze(0).to('cuda')
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                   return_dict_in_generate=True, output_scores=True)
    filling = tokenizer.batch_decode(generated_ids.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return filling, [float(generated_ids['scores'][e][0][i].detach().cpu()) for e, i in
                     enumerate(generated_ids['sequences'][0][-len(generated_ids['scores']):])], \
        generated_ids['sequences'][0][-len(generated_ids['scores']):].tolist()


def generation_step_next_token(p, name, code, model, tokenizer, stopping_criteria, fill_in_the_middle=False,  line=False):
    split_string = (name + "(")
    prompt = '[INST]Provide the function name that will replace the FILL_FUNCTION_NAME. The answer format is only the "FILL_FUNCTION_NAME".[INST]\n'+ p.replace(split_string, "FILL_FUNCTION_NAME(")+'[ANS]'
    print(prompt)
    generator = outlines.generate.text(model)
    answer = generator(prompt, max_tokens=30)
    
    print(answer)

    # breakpoint()
    # filling, scores, ids = generate(prompt, model, tokenizer, stopping_criteria)
    # print(prompt)
    # print(f'Ans: {filling} ')
    # print(f'real: {name}')

    # df.loc[len(df)] = {'prompt': prompt, 'function_name': name, 'generated': filling, 'scores': scores, 'ids': ids}



def process_row(ex, model, tokenizer, stopping_criteria, fill_in_the_middle=False, line=False):
    
    generation_step_next_token(ex['func_definition'],  ex['func_name'], ex['code'], model, tokenizer, stopping_criteria)
    df.to_csv(f'/home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')


def main():
    setup_logging()
    model, tokenizer = setup_model_and_tokenizer()

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=torch.tensor([13, 313]))])
    # Load data
    code_data = pd.read_csv(CODE_DATA_PATH, index_col=0).reset_index()

    logging.info(f"---------------------------\n\n")
    for j in tqdm(range(4)):#code_data.shape[0])):
        ex = code_data.loc[j]
        process_row(ex, model, tokenizer, stopping_criteria, fill_in_the_middle=True, line=True)
       
            
    logging.info(f"LLAMA instruct names generation")
    logging.info(
        f'Generation info saved to file /home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
    logging.info(f"Dataset size is: {df.shape}")

    # Remember to close the file handler to release the resources
    logging.shutdown()

if __name__ == "__main__":
    main()