from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import pandas as pd
from tqdm import tqdm
import logging
import json
import warnings
import time
warnings.filterwarnings("ignore")



logging.basicConfig(filename='/home/sasha/effective-inference/clean_naming/logs/accurasies.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", load_in_8bit=True,
    device_map="auto")

code_data = pd.read_csv('code_data.csv')

df = pd.DataFrame(columns = ['name_type', 'prompt', 'real', 'generated', 'answer'])
st = time.time()
acc_dict = {'all': 0, 'Original':0, 'GPT generated':0, 'Numerical': 0}

def generate_and_save(prompt, code, name, type):
    # prompt is based on huggingface example
    PROMPT = prompt + "\n" + (name+"(").join(code.split(name+"(")[:i-1]) + "<FILL_ME>"
    
    # generation
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to('cuda')
    generated_ids = model.generate(input_ids, max_new_tokens=10)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]

    # logging
    if filling[:len(name)] == name:
        acc_dict[type]+=1
    df.loc[len(df)] = {'name_type':type, 'prompt':PROMPT, 'real': (name+"(")+(name+"(").join(code.split(name+"(")[i-1:]), 'generated':filling, 'answer': filling[:len(name)] == name} 


for j in tqdm(range(2)):#code_data.shape[0]):
    ex = code_data.loc[j]
    prompt_names_dict = json.loads(ex['prompt_names_dict'])
    prompt_numerical_dict = json.loads(ex['prompt_numerical_dict'])
    for name, bad_name in prompt_names_dict.items():
        numerical_name = prompt_numerical_dict[name]
        for i in range(ex['prompt'].count(name+"(")):
            # print(name, bad_name, numerical_name, "\n\n")
            acc_dict['all']+=1

            generate_and_save(ex['prompt'], ex['code'], name, 'Original')
            
            generate_and_save(ex['bad_prompt'], ex['bad_code'], bad_name, 'GPT generated')

            generate_and_save(ex['numerical_prompt'], ex['numerical_code'], numerical_name, 'Numerical')
            
            df.to_csv(f'logs/generation_data_{st}.csv')

logging.info(f"---------------------------\n\n")
logging.info(f"Dataset size is: {acc_dict['all']}")
print(acc_dict)
print(f"Original functions: {acc_dict['Original']/acc_dict['all']}")
logging.info(f"Original functions: {acc_dict['Original']/acc_dict['all']}")
print(f"GPT generated functions: {acc_dict['GPT generated']/acc_dict['all']}")
logging.info(f"GPT generated functions: {acc_dict['GPT generated']/acc_dict['all']}")
print(f"Numerical functions: {acc_dict['Numerical']/acc_dict['all']}")
logging.info(f"Numerical functions: {acc_dict['Numerical']/acc_dict['all']}")

# Remember to close the file handler to release the resources
logging.shutdown()
        