from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import pandas as pd
from tqdm import tqdm
import logging
import json
import warnings
warnings.filterwarnings("ignore")



logging.basicConfig(filename='/home/sasha/effective-inference/accurasies.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", load_in_8bit=True,
    device_map="auto")

code_data = pd.read_csv('code_data.csv')

# logging.info('')

acc_dict = {'all': 0, 'original':0, 'full_changed_names':0, 'numerical_names': 0}
for j in tqdm(range(2)):#code_data.shape[0]):
    ex = code_data.loc[j]
    prompt_names_dict = json.loads(ex['prompt_names_dict'])
    prompt_numerical_dict = json.loads(ex['prompt_numerical_dict'])
    for name, bad_name in prompt_names_dict.items():
        numerical_name = prompt_numerical_dict[name]
        for i in range(ex['prompt'].count(name+"(")):
            # print(name, bad_name, numerical_name, "\n\n")
            acc_dict['all']+=1
            PROMPT = ex['prompt'] + "\n" + (name+"(").join(ex['code'].split(name+"(")[:i-1]) + "<FILL_ME>"
            # print("\nPrompt\n", PROMPT)
            input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to('cuda')
            generated_ids = model.generate(input_ids, max_new_tokens=10)
            filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
            if filling[:len(name)] == name:
                acc_dict['original']+=1

            BAD_PROMPT = ex['bad_prompt'] + "\n" + (bad_name+"(").join(ex['bad_code'].split(bad_name+"(")[:i-1]) + "<FILL_ME>"
            # print("\nPrompt\n", BAD_PROMPT)
            input_ids = tokenizer(BAD_PROMPT, return_tensors="pt")["input_ids"].to('cuda')
            generated_ids = model.generate(input_ids, max_new_tokens=10)
            filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
            if filling[:len(bad_name)] == bad_name:
                acc_dict['full_changed_names']+=1
                
            NUMERICAL_PROMPT = ex['numerical_prompt'] + "\n" + (numerical_name+"(").join(ex['numerical_code'].split(numerical_name+"(")[:i-1]) + "<FILL_ME>"
            # print("\nPrompt\n",NUMERICAL_PROMPT)
            input_ids = tokenizer(NUMERICAL_PROMPT, return_tensors="pt")["input_ids"].to('cuda')
            generated_ids = model.generate(input_ids, max_new_tokens=10)
            filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
            if filling[:len(numerical_name)] == numerical_name:
                acc_dict['numerical_names']+=1






print(f"Original functions: {acc_dict['original']/acc_dict['all']}")
logging.info(f"Original functions: {acc_dict['original']/acc_dict['all']}")
logging.info(f"Original function promt example: {PROMPT}")
print(f"GPT generated functions: {acc_dict['full_changed_names']/acc_dict['all']}")
logging.info(f"GPT generated functions: {acc_dict['full_changed_names']/acc_dict['all']}")
logging.info(f"GPT generated function promt example: {BAD_PROMPT}")
print(f"Numerical functions: {acc_dict['full_changed_names']/acc_dict['all']}")
logging.info(f"Numerical functions: {acc_dict['full_changed_names']/acc_dict['all']}")
logging.info(f"Numerical function promt example: {NUMERICAL_PROMPT}")

# Remember to close the file handler to release the resources
logging.shutdown()
        