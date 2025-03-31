"""
Script to test if DNABERT works on the environment 
"""
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel
# import pkg_resources
import einops
import peft
import omegaconf
import evaluate
import accelerate
#import triton

# print(pkg_resources.get_distribution("transformers").version)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('CUDA AVAILABLE')
else:
    print('CUDA NOT AVAILABLE')

print('downloading tokenizer')
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
print('downloading model')
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
model = model.to(device)

print('proceeding to example task')
dna = "[PAD]"#"GAAT[PAD]AAGCTAA"
print('dna seq len: ' ,len(dna))


inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
#inputs = tokenizer.batch_encode_plus(dna, padding=True, return_attention_mask=True, return_tensors = 'pt')
#inputs2 = tokenizer.encode(dna2, padding=True, return_tensors='pt')
print('Tokenized sequences:\n{}'.format(inputs))
print('Decoded:\n{}'.format(tokenizer.batch_decode(inputs['input_ids'])))#, tokenizer.decode(inputs2['input_ids'])))

#inputs = {k: v.to(device) for k, v in inputs.items()}
print('hidden states')
hidden_states = model(inputs.to(device))[0]
#hidden_states = model(**inputs)[0] # [1, sequence_length, 768]

print('embedding')
# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768
#print(embedding_mean.to('cpu')[0])
# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape) # expect to be 768

print('done')
