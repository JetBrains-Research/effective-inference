from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import pandas as pd
from tqdm import tqdm
import logging
import json
import warnings
import re
import time
warnings.filterwarnings("ignore")



logging.basicConfig(filename='/home/sasha/effective-inference/clean_naming/logs/accurasies.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", load_in_8bit=True,
    device_map="auto")

code_data = pd.read_csv('/home/sasha/effective-inference/clean_naming/code_data.csv')

df = pd.DataFrame(columns = ['name_type', 'prompt', 'function_name', 'real', 'generated', 'answer'])
st = time.time()
acc_dict = {'all': 0, 'Original':0, 'GPT generated':0, 'Numerical': 0}

def generate(PROMPT, max_new_tokens = 10):
    # generation
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to('cuda')
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]

    return filling
    # logging
    

def process_row_next_token(ex, name, bad_name, numerical_name, translit_name):
    for i in range(ex['prompt'].count(name+"(")):
            # print(name, bad_name, numerical_name, "\n\n")
            acc_dict['all']+=1

            prompt = ex['prompt'] + "\n" + (name+"(").join(ex['code'].split(name+"(")[:i-1]) + "<FILL_ME>"
            real = (name+"(")+(name+"(").join(ex['code'].split(name+"(")[i-1:])
            filling = generate(prompt)
            if filling[:len(name)] == name:
                acc_dict['Original']+=1
            df.loc[len(df)] = {'name_type':'Original', 'prompt':prompt, 'function_name':name, 'real': real, 'generated':filling, 'answer': filling[:len(name)] == name} 


            prompt = ex['bad_prompt'] + "\n" + (bad_name+"(").join(ex['bad_code'].split(bad_name+"(")[:i-1]) + "<FILL_ME>" 
            real = (bad_name+"(")+(bad_name+"(").join(ex['bad_code'].split(bad_name+"(")[i-1:])
            filling = generate(prompt)
            if filling[:len(bad_name)] == bad_name:
                acc_dict['GPT generated']+=1
            df.loc[len(df)] = {'name_type':'GPT generated', 'prompt':prompt, 'function_name':bad_name, 'real': real, 'generated':filling, 'answer': filling[:len(bad_name)] == bad_name} 

        
            prompt = ex['numerical_prompt']+ "\n" + (numerical_name+"(").join(ex['numerical_code'].split(numerical_name+"(")[:i-1]) + "<FILL_ME>"
            real = (numerical_name+"(")+(numerical_name+"(").join(ex['numerical_code'].split(numerical_name+"(")[i-1:])
            filling = generate(prompt)
            if filling[:len(numerical_name)] == numerical_name:
                acc_dict['Numerical']+=1
            df.loc[len(df)] = {'name_type':'Numerical', 'prompt':prompt, 'function_name':numerical_name, 'real': real, 'generated':filling, 'answer': filling[:len(numerical_name)] == numerical_name} 

            prompt = ex['translit_prompt']+ "\n" + (translit_name+"(").join(ex['translit_code'].split(translit_name+"(")[:i-1]) + "<FILL_ME>"
            real = (translit_name+"(")+(translit_name+"(").join(ex['translit_code'].split(translit_name+"(")[i-1:])
            filling = generate(prompt)
            if filling[:len(translit_name)] == translit_name:
                acc_dict['Translit']+=1
            df.loc[len(df)] = {'name_type':'Translit', 'prompt':prompt, 'function_name':translit_name, 'real': real, 'generated':filling, 'answer': filling[:len(translit_name)] == translit_name} 

            
            df.to_csv(f'/home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
        

def process_row_line(ex, name, bad_name, numerical_name, translit_name):
    lines = ex['code'].split('\n')
    matching_lines = [i for i, line in enumerate(lines) if re.search(fr"{name}\(", line)]
    max_new_tokens = 50
    for i in matching_lines:
        # print(name, bad_name, numerical_name, "\n\n")
        acc_dict['all']+=1
        prompt = ex['prompt']+"\n"+ "\n".join(ex['code'].split('\n')[:i])+"\n<FILL_ME>"
        if i+1 < len(lines):
            prompt += "\n"+"\n".join(ex['code'].split('\n')[i+1:])
        real = ex['code'].split('\n')[i]
        filling = generate(prompt)
        if name+"(" in filling:
            acc_dict['Original']+=1
        df.loc[len(df)] = {'name_type':'Original', 'prompt':prompt, 'function_name':name,'real': real, 'generated':filling, 'answer': (name+"(" in filling)} 

        
        prompt = ex['bad_prompt']+"\n"+ "\n".join(ex['bad_code'].split('\n')[:i])+"\n<FILL_ME>"
        if i+1 < len(lines):
            prompt += "\n"+"\n".join(ex['bad_code'].split('\n')[i+1:])
        real = ex['bad_code'].split('\n')[i]
        filling = generate(prompt)
        if bad_name+"(" in filling:
            acc_dict['GPT generated']+=1
        df.loc[len(df)] = {'name_type':'GPT generated', 'prompt':prompt,'function_name':bad_name, 'real': real, 'generated':filling, 'answer': (bad_name+"(" in filling)}

        
        prompt = ex['numerical_prompt']+"\n"+ "\n".join(ex['numerical_code'].split('\n')[:i])+"\n<FILL_ME>"
        if i+1 < len(lines):
            prompt += "\n"+"\n".join(ex['numerical_code'].split('\n')[i+1:])
        real = ex['numerical_code'].split('\n')[i]
        filling = generate(prompt, max_new_tokens)
        if numerical_name+"(" in filling:
            acc_dict['Numerical']+=1
        df.loc[len(df)] = {'name_type':'Numerical', 'prompt':prompt, 'function_name':numerical_name,'real': real, 'generated':filling, 'answer': (numerical_name+"(" in filling)}

        prompt = ex['translit_prompt']+"\n"+ "\n".join(ex['translit_code'].split('\n')[:i])+"\n<FILL_ME>"
        if i+1 < len(lines):
            prompt += "\n"+"\n".join(ex['translit_code'].split('\n')[i+1:])
        real = ex['translit_code'].split('\n')[i]
        filling = generate(prompt, max_new_tokens)
        if translit_name+"(" in filling:
            acc_dict['Translit']+=1
        df.loc[len(df)] = {'name_type':'Translit', 'prompt':prompt, 'function_name':translit_name,'real': real, 'generated':filling, 'answer': (translit_name+"(" in filling)}

        df.to_csv(f'/home/sasha/effective-inference/clean_naming/logs/generation_data_{st}.csv')
    


for j in tqdm(range(code_data.shape[0])):
    ex = code_data.loc[j]
    prompt_names_dict = json.loads(ex['prompt_names_dict'])
    prompt_numerical_dict = json.loads(ex['prompt_numerical_dict'])
    for name, bad_name in prompt_names_dict.items():
        numerical_name = prompt_numerical_dict[name]
        translit_name = translit_names_dict[name]
        process_row_line(ex, name, bad_name, numerical_name, translit_name)
    
    if j%50==0:
            print(f"iteration {j} results: {acc_dict}")

logging.info(f"---------------------------\n\n")
logging.info(f"Dataset size is: {acc_dict['all']}")
print(acc_dict)
print(f"Original functions: {acc_dict['Original']/acc_dict['all']}")
logging.info(f"Original functions: {acc_dict['Original']/acc_dict['all']}")


print(f"GPT generated functions: {acc_dict['GPT generated']/acc_dict['all']}")
logging.info(f"GPT generated functions: {acc_dict['GPT generated']/acc_dict['all']}")


print(f"Numerical functions: {acc_dict['Numerical']/acc_dict['all']}")
logging.info(f"Numerical functions: {acc_dict['Numerical']/acc_dict['all']}")

print(f"Translit functions: {acc_dict['Translit']/acc_dict['all']}")
logging.info(f"Translit functions: {acc_dict['Translit']/acc_dict['all']}")

logging.info(f"Mean len of original functions: {df[df['name_type']=='Original']['function_name'].apply(len).mean()}")
logging.info(f"Mean len of GPT generated functions: {df[df['name_type']=='GPT generated']['function_name'].apply(len).mean()}")
logging.info(f"Mean len of Numerical functions: {df[df['name_type']=='Numerical']['function_name'].apply(len).mean()}")
logging.info(f"Mean len of Translit functions: {df[df['name_type']=='Translit']['function_name'].apply(len).mean()}")

print(f"Mean len of original functions: {df[df['name_type']=='Original']['function_name'].apply(len).mean()}")
print(f"Mean len of GPT generated functions: {df[df['name_type']=='GPT generated']['function_name'].apply(len).mean()}")
print(f"Mean len of Numerical functions: {df[df['name_type']=='Numerical']['function_name'].apply(len).mean()}")
print(f"Mean len of Translit functions: {df[df['name_type']=='Translit']['function_name'].apply(len).mean()}")

# Remember to close the file handler to release the resources
logging.shutdown()
        