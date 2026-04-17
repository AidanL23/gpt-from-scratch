#download the dataset
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "input.txt"
)
#read it in
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))
print(text[:1000])

#get sorted list of all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))

#character int mapping
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: takes a string, output a list of ints
decode = lambda l: ''.join([itos[i]for i in l]) #decoder: take a list of ints, output a string

# Test it
print(encode("hi there"))
print(decode(encode("hi there")))

#encoding the entire text dataset and stores it into torch.tensor
import torch # uses PyTorch: https://pytorch.org

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) #dispalys the first 1000 characters just encoded 

#spliting data into trian and validation sets
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8 #maximum context length for preditions
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences it will be processing
block_size = 8 # the max context length for predictions

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.radint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size] for i in ix])
    return x, y

    xb, yb = get_batch('train')
    print('inputs')
    print(xb.shapes)
    print(xb)
    print('targets')
    print(yb.shape)
    print(yb)

    print('----')

    for b in range(batch_size): #bathc dimension
        for t in range(block_size): #time dimension
            context = xb[b, :t+1] #the input characters up to the current time step
            target = yb[b, t] #the target character at the current time step
            print(f"when input is {context.tolist()} the target: {target}")

