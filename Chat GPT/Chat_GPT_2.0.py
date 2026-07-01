import torch
import pickle
import torch.nn.functional as F
import torch.nn as nn
import argparse
import random
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
def parse_args():
  parse = argparse.ArgumentParser(description="Example")
  parse.add_argument("-batch_size", type=int,required=True, help = "Provide the batch_size")
  return parse.parse_args
args = parse_args()
print(f"Batch_size: {args.batch_size}")
block_size = 64
batch_size = 128
n_embd = 384
learning_rate = 3e-4
eval_iter = 250
max_iters = 3000
n_layers = 8
n_heads = 8
dropout = 0.2

chars  = ""
with open("/content/vocab.txt",  "r" , encoding = "utf-8") as file: #START TO TRAIN IT ON THE ETHERNET INFO
  text = file.read()
  chars = sorted(list(set(text)))
  vocab = len(chars)
  string_to_int = {ch:i for i,ch in enumerate(chars)}
  int_to_string = {i:ch for i,ch in enumerate(chars)}
  encode = lambda s: [string_to_int[c] for c in s]
  decode = lambda l: "".join([int_to_string[v] for v in l])

  data = torch.tensor(encode(text), dtype = torch.long)

def get_random_chunk(split):
  filename = "/content/output_train.txt" if split == "train" else "/content/output_valid.txt"
  with open(filename, "rb") as f:
    with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
      file_size  = len(mm)
      start_pos = random.randint(0, (file_size) - block_size*batch_size)
      mm.seek(start_pos)
      block = mm.read(block_size*batch_size-1)
      decoded_block  = block.decode("utf-8", errors = "ignore").replace("\n", " ")
      data = torch.tensor(encode(decoded_block), dtype = torch.long)
  return data

def get_batch(split):
  data = get_random_chunk(split)
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i + block_size] for i in ix ])
  y = torch.stack([data[i + 1: i + block_size + 1  ] for i in ix])
  x = x.to(device)
  y = y.to(device)
  return x,y

n = int(0.8 * len(data))
train_data = data[:n]
valid_data = data[n:]

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    #input of size (bacth, time-step, channels)
    #output of size (batch, time-step, head_size)
    B,T,C = x.shape
    k = self.key(x) #(B,T, hs)
    q = self.query(x) #(B,T, hs)

    #computatioon
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,T,hs) @ (B, hs, T) --> (B,T,T) / hs
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf') )
    wei = F.softmax(wei, dim = -1) # (B,T(Key),T(Query))
    wei = self.dropout(wei)
    #perform the weighted agrregation of the values
    v = self.value(x)
    out = wei @ v
    return out







class MultiheadAttention(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(head_size * n_heads, n_embd )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out





class FeedForward(nn.Module):
 def __init__(self, n_embd):
  super().__init__()
  self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
  )

 def forward(self, x):
  return self.net(x)



class Block(nn.Module):
  def __init__(self, n_embd, n_heads):
   super().__init__()
   head_size = n_embd // n_heads
   self.sa = MultiheadAttention(n_heads, head_size)
   self.ffwd = FeedForward(n_embd)
   self.ln1 = nn.LayerNorm(n_embd)
   self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
   #x = x + self.sa(self.ln1(x))
   #x = x + self.ffwd(self.ln2(x))
   y = self.sa(x)
   x = self.ln1(x+y)
   y = self.ffwd(x)
   x = self.ln2(x + y)
   return x




class GPTlanguageModel(nn.Module):
  def __init__(self, vocab):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_heads = n_heads) for _ in range(n_layers)])

    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

  def forward(self, index, targets = None ):

    B,T = index.shape
    tok_embd = self.token_embedding_table(index)
    pos_embd = self.position_embedding_table(torch.arange(T, device = device))
    x = tok_embd + pos_embd
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)


    if targets is None:
      loss = None
    else:
      B, T , C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, index,max_tokens):
    for _ in range(max_tokens):
      index_cond = index[:, -block_size:]
      logits, loss = self.forward(index_cond)
      logits = logits[:, -1, :]

      probs = F.softmax(logits, dim = -1 )

      index_next = torch.multinomial(probs, num_samples  = 1 )
      index = torch.cat((index, index_next), dim = 1 )
    return index

model = GPTlanguageModel(vocab)
print("Loading the model parameters.....")
#with open("model-01.pkl", "rb") as f: #After getting a pre-trained model!!!
  #pickle.load(f)
 # print("Model has been downloaded successfuly")
m = model.to(device)

@torch.no_grad()
def estemate_loss():
  out = {}
  model.eval()
  for split in ['train', 'valid']:
    losses = torch.zeros(eval_iter)
    for k in range(eval_iter):
      X, Y  = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
for iter in range(max_iters):
  if iter % eval_iter == 0:
    losses = estemate_loss()
    print(f"step {iter}, train losses = {losses['train']:.3f}, valid losses = {losses['valid']:.3f} ")
  xb, yb = get_batch('train')
  logits, loss = model.forward(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
print(loss.item())
#with open("model-01.pkl", "wb") as f: #save your pre-trained model
  #pickle.dump(model, f)

while True:
  prompt = input("Prompt: ") #in order to chat with chat-gpt
  context = torch.tensor(encode(prompt), dtype = torch.long, device = device)
  generated_text = decode(m.generate(context, max_tokens=150)[0].tolist())
  print(generated_text)

