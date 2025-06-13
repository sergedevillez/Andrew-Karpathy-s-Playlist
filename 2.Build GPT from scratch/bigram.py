import torch
import torch.nn as nn
from torch.nn import functional as F


# This is a DECODER-only transformer. Normaly a encoder would take the input and encode it into a fixed-size representation, which is then passed to a decoder that generates the output. In this case, we are only interested in generating text, so we only need the decoder part of the transformer.

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # how many training iterations to run?
eval_interval = 500 # how often to evaluate the loss on the validation set?
learning_rate = 3e-4 # the initial learning rate for Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu' # allow the use of the GPU with Cuda if available
eval_iters = 200 # how many batches to use for the evaluation set?
n_embedding_dimensions = 384 # the number of dimensions for the token embeddings
n_head = 6
n_layer = 6
dropout = 0.2 # dropout rate to prevent overfitting (shut down random neurons during each training to prevent overfitting)
# ------------


torch.manual_seed(1337) # for reproducibility

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# here are all the unique characters that occurs in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers and back
stoi = { ch: i for i, ch in enumerate(chars) } # string to integer
itos = { i: ch for i, ch in enumerate(chars) } # integer to string
# Encoder - decoder
encode = lambda s: [stoi[c] for c in s] # encoder: take a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers and output a string

# split data into training and validation data
data = torch.tensor(encode(text), dtype=torch.long) 
n = int(0.9 * len(data)) # first 90% for training, rest for validation
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices
    x = torch.stack([data[i:i + block_size] for i in ix]) # input batch
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]) # target batch
    x, y = x.to(device), y.to(device) # move to the appropriate device (CPU or GPU)
    return x, y


# this function average the loss among multiple batches, lot less noisy than checking the loss on a single batch
@torch.no_grad()
def estimate_loss():
    out = {}
    # currently, changing the mode doesn't serve a use but in theory
    # setting the model to evaluation mode, this disables dropout and batch normalization
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # setting the model back to training mode, this enables dropout and batch normalization
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    ''' Multi-head self-attention layer  running in parralel'''

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embedding_dimensions) # linear layer to project the concatenated outputs of all heads back to the embedding dimension
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # apply each head to the input and concatenate the outputs along the channel dimension
        out = self.proj(out) # project the concatenated output back to the embedding dimension
        return out

class Head(nn.Module):
    ''' A single head of self-attention '''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(n_embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(n_embedding_dimensions, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix for causal attention

        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

    def forward(self, x):
        B, T, C = x.shape
        # calculate the key, query, and value vectors
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # apply the causal mask
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # apply dropout to the attention weights
        # performe the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embedding_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding_dimensions, 4 * n_embedding_dimensions), # linear layer
            nn.ReLU(), # non-linearity
            nn.Linear(4 * n_embedding_dimensions, n_embedding_dimensions), # another linear layer
            nn.Dropout(dropout), # added right before the residual connection connect back to the pathway
        )

    def forward(self, x): # is done per token 
        return self.net(x) # apply the feed-forward network

class Block(nn.Module):
    """ A single transformer block consisting of self-attention and feed-forward network """

    def __init__(self, n_embedding_dimensions, n_head=4):
        super().__init__()
        head_size = n_embedding_dimensions // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embedding_dimensions)
        self.ln1 = nn.LayerNorm(n_embedding_dimensions) # layer normalization after self-attention
        self.ln2 = nn.LayerNorm(n_embedding_dimensions) # layer normalization after feed-forward network

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # apply self-attention
        x = x + self.ffwd(self.ln2(x)) # apply feed-forward network
        return x

# Super simple Bigram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding_dimensions)
        self.blocks = nn.Sequential(*[Block(n_embedding_dimensions, n_head=n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embedding_dimensions) # final layer normalization
        self.lm_head = nn.Linear(n_embedding_dimensions, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape # B is the batch size, T is the sequence length

        # idx and targets are both (B, T) tensor of integers
        token_embd = self.token_embedding_table(idx) # (B, T, C) where C is the number of embedding dimensions
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # integer from 0 to T-1, shape (T, C)
        x = token_embd + pos_emb # (B, T, C) where C is the number of embedding dimensions
        x = self.transformer_blocks(x) # apply transformer blocks, shape remains (B, T, C)
        logits = self.lm_head(x) # (B, T, C) where C is the number of classes (vocab_size)

        if(targets is None):
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshape to (B*T, C) because Torch expects only two dimensions and the last dimension to be the channel's dimension
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) tensor of integers
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim =-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the runnin sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
# move the model back to the device
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    # backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device) # start a piece of context on the device
print((decode(m.generate(context, max_new_tokens=500)[0].tolist())))
