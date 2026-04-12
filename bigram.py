#download the dataset
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "data/input.txt"
)
#read it in
with open('input.txt', 'r', encodings='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))
print(text[:1000])

#get sorted list of all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))

#character int mapping
stoi = { ch:1 for i, ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: takes a string, output a list of ints
decode = lambda l: ''.join([itos[i]for i in l]) #decoder: take a list of ints, output a string

# Test it
print(encode("hi there"))
print(decode(encode("hi there")))