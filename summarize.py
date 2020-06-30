#!python3
from transformers import *
import sys

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)

from summarizer import Summarizer
path = sys.argv[1]
f = open(path,"r")
body = f.read()
f.close()
model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
print(full)