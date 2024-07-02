import csv
import os
from contextlib import nullcontext
import torch
import tiktoken
import numpy as np
import random
import json
import math
import os
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
import calendar
from tinystories import get_tokenizer_model_path

#prefix
upvotes = 200
location = "USA"
# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

def get_ppl(model, start, starts_with):
    start_ids = enc.encode(starts_with+start, bos=True, eos=False)
    x = (torch.tensor(start_ids[:-1], dtype=torch.long, device=device)[None, ...])
    y = (torch.tensor(start_ids[1:], dtype=torch.long, device=device)[None, ...])
    # run generation
    with torch.no_grad():
        with ctx:
            logits, loss = model(x, y)
            return loss.exp().item()

def get_log_probability(model, text, starts_with, device):
    start_ids = enc.encode(starts_with + text, bos=True, eos=False)
    x = torch.tensor(start_ids[:-1], dtype=torch.long, device=device)[None, ...]
    y = torch.tensor(start_ids[1:], dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            logits, _ = model(x, y)
    
    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
    total_log_prob = token_log_probs.sum().item()
    return total_log_prob

def score_file_peak(filename, model):
   score_right = 0
   score_all = 0
   # initializing the titles and rows list
   rows = []

   with open(dir_path+'/'+filename, 'r') as csvfile:
       # creating a csv reader object
       csvreader = csv.reader(csvfile)

       # extracting each data row one by one
       for row in csvreader:
           rows.append(row)
   for row in rows:
       peak_month = row[11]

       # parsing each column of a row
       for col in row[:11]:
           countOfWords = len(col.split())
           if len(col)==0 or countOfWords <=1:
               continue

           print(col)

           ppls = []
           for i in range(1,13):
               month_text = calendar.month_name[i].upper()
               starts_with = "\n[YEAR: 2023 MONTH: "+ month_text + " SOURCE: NYT_USERS UPVOTES: " + str(
                        upvotes) + " LOCATION: " + location + "] "
               ppl = get_ppl(model, col, starts_with)
               ppls.append(ppl)

           min_index = ppls.index(min(ppls))+1
           print("our_peak_month: "+str(min_index))
           print("official_peak_month: "+str(peak_month))
           if str(min_index) == peak_month:
               score_right+=1
               score_all+=1
               print('Its a catch!')
               print(score_right / score_all)
           else:
               score_all += 1
           print()

   return score_right/score_all

def score_file_stability(filename, model):
   rows = []
   rstd_cum = 0
   rstd_count = 0
   # reading csv file
   with open(dir_path+'/'+filename, 'r') as csvfile:
       csvreader = csv.reader(csvfile)

       # extracting each data row one by one
       for row in csvreader:
           rows.append(row)

   for row in rows:

       # parsing each column of a row
       for col in row[:11]:
           countOfWords = len(col.split())
           if len(col)==0 or countOfWords <=1:
               continue

           print(col)

           ppls = []
           for i in range(1,13):
               month_text = calendar.month_name[i].upper()
               starts_with = "\n[YEAR: 2023 MONTH: "+ month_text + " SOURCE: NYT_USERS UPVOTES: " + str(
                        upvotes) + " LOCATION: " + location + "] "
               ppl = get_ppl(model, col, starts_with)
               ppls.append(ppl)

           avg = np.average(ppls)
           std = np.std(ppls)
           rstd = std/avg
           print(rstd)
           if not math.isnan(rstd):
               rstd_cum+=rstd
               rstd_count+=1

   return rstd_cum/rstd_count

# dictionary with list object in values
all_dict = {
}

# peak
acc = score_file_peak('peak_month.csv', models, jsons)
print('peak_month'+': '+str(acc))
all_dict['all']=acc

#stability
stability = score_file_stability('stability.csv', models, jsons)
print('stability'+': '+str(stability) )
all_dict['stability']=stability

print('stability'+': '+str(stability) )
print('acc'+': '+str(acc))

with open(model_sufix+'.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
   w = csv.DictWriter(f, all_dict.keys())
   w.writeheader()
   w.writerow(all_dict)

