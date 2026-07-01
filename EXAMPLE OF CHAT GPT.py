import torch
import time
device = "cuda" if torch.cuda.is_available() else "cpu" #us cuda in priority cause of paralel computations!
print(device) #check it
block_size = 8 # number of random sentences from the text
batch_size = 4 #number of letters from the random sentences from the text
learning_rate = 3e-4 #how fast will our machine learn smth
max_iters = 3000 #how many times(blocks) we need to train it
eval_iter = 500 #on which iter(block of train) we need to estimate the loss

with open("Your_textbook.txt", "w", encoding = "utf-8") as file: #open our text
  text = file.read() #read it
  chars = sorted(set(text)) #sort every new unique letter from the text
  vocab_size = len(chars) #the length of our vocabulary
  print(text[:100]) #print our first letters from the text

  string_to_int = {ch:i for i,ch in enumerate(chars)} # add a number for each character
  int_to_string = {i:ch for i,ch in enumerate(chars)} # add a character for each number


  encode = lambda s: [string_to_int[c] for c in s] #every letter we convert into integer(into number)
  decode = lambda l: "".join([int_to_string[h] for h in l ]) #every encoded letter we convert into (casual for people) letter


  data = torch.tensor(encode(text), dtype = torch.long) #make an encode text


  print(f"\n {data[:100]}") #print it in order to compare with the printed text


n = int(0.9 * len(data)) #in oreder to devide the text into 10% and 90%
train_data = data[:n] #train data is the first 90% of the letters
valid_data = data[n:] #validation data is the rest of the text(10%) in order to give our model exam

#block_size = 8;
x = train_data[:block_size] #current letter(context)
y = train_data[1:block_size + 1] #target
for t in range(block_size):
  context = x[:t +  1]
  target =  y[t]
  print("when input is: ", context, "target is: ", target)


#...[67,52,45,87]35... [:block_size] = [:4]
#...67[52,45,87,35].... [1:block_size + 1]
#h -> e
#e -> l
#l -> l
#l -> o

#hell
#ello
#hello

def get_batch(split):
  data = train_data if split == 'train' else valid_data
  ix = torch.randint(len(data) - block_size, (batch_size, )) #take a random piece of text in order to train on it
  x = torch.stack([data[i: i + block_size] for i in ix]) #stack the context
  y = torch.stack([data[i + 1: i + block_size + 1 ] for i in ix]) #stack the targets for the context
  x = x.to(device) #run this with GPU
  y = y.to(device) #run this with CPU
  return x,y

class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #make the table of vectors
    self.position_embedding_table =  nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range (n_layers)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
  def forward(self,index, targets = None):             #we don't have targets at the beginning in this model
    #logits = self.token_embedding_table(index)

    tok_embd = self.token_embedding_table(index)
    pos_embd = self.position_embedding_table(torch.arange (T, device = device))
    x = tok_embd + pos_embd
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None

    else:
      B, T , C = logits.shape
      logits  = logits.view(B*T, C) # B*T (Batch_size - how many times we took our random characters ("our future sentences") * charachters ("letters in this sentence "))
                                    # by multiplying these dimensions we got: before [44,55,66]  after [44,55,66,12,23,56]
                                    #                                                [12,23,56]
                                    # from 3D -----> 2D Array
                                    # Because Function:   'F.cross_entropy' can only analize 2D arrays
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets) #analize the right answers between our targets and change the value of weights: (B*T)

    return logits, loss

  def generate(self,index,max_new_tokens):
    for _ in range(max_new_tokens):
      logits,loss = self.forward(index)
      logits = logits[:, -1, :] # T - our symbols or charachters ("letters"), so we need to start from the end of our encode sentence by giving T = -1
                                # If T was 0, we would start to choose the best option ("which letter to choose next?") for the first letter and we will "ALWAYS" compare with it
                                #If T is -1, we start to choose the best option ("which letter to choose next?") for the 'EVERY' last letter in the each sentence, because our letter
                                # every time is various, cause we choose for every last letter ("what letter is gonna be next?")

                                #IF 0: WE HAVE FOR EXAMPLE WORD 'HEL': HEL ---> K ---> K ---> G ----> J ---> K ---> J AND SO ON....... THE WORD IS: HELKKGJKJ - INCORRENT
                                # AFTER H for example usually goes K, G, J

                                #IF -1: WE HAVE FOR EXAMPLE WORD 'HEL': HEL ---> L ----> O ---> '' AND SO ON........ THE WORD IS: HELL0 - CORRECT
                                #AFTER L usually goes another L or O, then after L like i said before goes O too, After O ussually we have 'space'

                                # WHY 0 - beginning and -1 is the last letter
                                #  [1,2,3]
                                # [0] = 1, [1] = 2, [2] = 3
                                #  <---(-1)[1,2,3]<------ so that's why [2] = [-1] = 3
                                # it's beneficial when we do not know an amount of  letters of the sentences or don't care about them, but actually always can start count from the last letter and rely on it!


      probs = F.softmax(logits, dim = -1) # get our precentage of probabilities for every last letter
      index_next = torch.multinomial(probs, num_samples = 1)
      index = torch.cat((index, index_next), dim = 1) #summarise every letter together
    return index


model  = GPTLanguageModel(vocab)
m = model.to(device)
context = torch.zeros((1,1), dtype = torch.long, device = device) # we start from the 0 index
                                                                  # torch.zeros(1,1) the first number is our batch_size (how many sentence do we need to generate ), the second number is how many numbers at the beginning we have (we have the first and only one number: "\n")
generated_chars = decode(m.generate(context, max_new_tokens= 500)[0].tolist()) #decode all the numbers into symbols or letters
print(generated_chars)
#CHAT GPT EXAMPLE (USE THIS!!!)

@torch.no_grand()
def estemate_eval():
  out = {}
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(eval_iter)
    for k in range(eval_iter):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()

  model.train()
  return out

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
for iters in range(max_iters):
  if iters % eval_iter == 0:
    losses = estemate_eval()
    print("step: ", iters, "losses: ", losses)
    print(f"step: {iters}, train losses: {losses['train']}, valid losses {losses['valid']} "  )
  xb, yb = get_batch("train")
  logits, loss = model.forward(xb, yb)
  optimizer.zero_grand(set_to_none = True)
  loss.backward()
  optimizer.step()
  losses = loss.item

context = torch.zeros((1,1), dtype = torch.long, device = device)
generated_text = decode(m.generate(context, max_new_tokens = 500)[0].tolist())
print(generated_text)
  


