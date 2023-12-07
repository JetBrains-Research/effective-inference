from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import pandas as pd
from tqdm import tqdm

tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", load_in_8bit=True,
    device_map="auto")

func_data = pd.read_csv('functions_data.csv')
counter = 0
counter_orig = 0
for i in tqdm(range(func_data.shape[0])):
    ex = func_data.loc[i]
    print(ex['func_name'], "  ->  ", ex['bad_name'])
    
    for i in range(int(ex['count'])):
        PROMPT = ex['prompt'].replace(ex['func_name'], ex['bad_name'])+"\n"+ex['bad_name'].join(ex['code'].replace(ex['func_name'], ex['bad_name']).split(ex['bad_name'])[:i-1])+"<FILL_ME>"
        input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to('cuda')
        generated_ids = model.generate(input_ids, max_new_tokens=10)

        filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
        # print(f"???{filling}???\n!!!{ex['bad_name']+ex['code'].replace(ex['func_name'], ex['bad_name']).split(ex['bad_name'])[i-1][:20]}!!!")
        
        if filling[:len(ex['bad_name'])] == ex['bad_name']:
            print('horey!')
            counter+=1
        else: pass

        PROMPT = ex['prompt']+"\n"+ex['bad_name'].join(ex['code'].split(ex['func_name'])[:i-1])+"<FILL_ME>"
        input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to('cuda')
        generated_ids = model.generate(input_ids, max_new_tokens=10)

        filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
        # print(f"???{filling}???\n!!!{ex['bad_name']+ex['code'].replace(ex['func_name'], ex['bad_name']).split(ex['bad_name'])[i-1][:20]}!!!")
        
        if filling[:len(ex['func_name'])] == ex['func_name']:
            # print('horey!')
            counter_orig+=1
        else: pass
print(f"Acc: {counter/func_data['count'].sum()}")
print(f"Acc orig: {counter_orig/func_data['count'].sum()}")
        
        