import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

pd.set_option('max_colwidth', 200)

df = pd.read_csv('shortjokes.csv', index_col=0, dtype=str)
df.tail(3)

from nltk.tokenize import TweetTokenizer
tk = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
df['tokenized'] = df.Joke.map(tk.tokenize)

BOS, EOS = ' ', '\n'

lines = df.Joke.apply(lambda line: BOS + line.replace(EOS, ' ') + EOS).tolist()

from tqdm import tqdm_notebook

d = {}
tokens = []

for line in tqdm_notebook(lines):
    for token in list(line):
        if token not in d: 
            d[token] = 1
            tokens.append(token)
        else: d[token] += 1

tokens = sorted(tokens)
n_tokens = len(tokens)
print ('n_tokens = ',n_tokens)

assert BOS in tokens, EOS in tokens

token_to_id = {}
for ind, elem in enumerate(tokens):
    token_to_id[elem] = ind

list(token_to_id.items())[:5]

def to_matrix(lines, max_len=None, pad=token_to_id[EOS], dtype='int32'):
    """Casts a list of lines into tf-digestable matrix"""
    max_len = max_len or max(map(len, lines))
    lines_ix = np.zeros([len(lines), max_len], dtype) + pad
    for i in range(len(lines)):
        line_ix = list(map(token_to_id.get, lines[i][:max_len]))
        lines_ix[i, :len(line_ix)] = line_ix
    return lines_ix

dummy_lines = [
    ' A cow walks into a bar...\n',
    ' What do you call a pirat on a plain?\n'
]
print(to_matrix(dummy_lines))



class RNNLanguageModel(nn.Module):
    def __init__(self, n_tokens = n_tokens, emb_size=16, hid_size=512, gpu = -1):
        super(RNNLanguageModel, self).__init__()
        """ 
        Build a recurrent language model.
        """
        self.gpu = gpu
        self.emb = nn.Embedding(n_tokens, emb_size)        
        self.lstm = nn.LSTM(input_size = emb_size, hidden_size = hid_size, bidirectional = False, batch_first = True) 
        self.dense = nn.Linear(hid_size, n_tokens)        
        self.next_token_probs = nn.Softmax(dim=2)

        
    
    def forward(self, input_ix):
        """
        compute language model logits given input tokens
        :param input_ix: batch of sequences with token indices, tf tensor: int32[batch_size, sequence_length]
        :returns: pre-softmax linear outputs of language model [batch_size, sequence_length, n_tokens]
            these outputs will be used as logits to compute P(x_t | x_0, ..., x_{t - 1})
        """
        
        emb_ix = self.emb(input_ix)       
        output, hidden = self.lstm(emb_ix)     
        logits_ix = self.dense(output)
        return logits_ix

    def get_possible_next_tokens(self, prefix=BOS, temperature=1.0, max_len=100):
        """ :returns: probabilities of next token, dict {token : prob} for all tokens """
        prefix_tensor = torch.from_numpy(to_matrix([prefix])).to(torch.int64)
        prefix_tensor = to_gpu(prefix_tensor, self.gpu)        
        probs = self.next_token_probs (self(prefix_tensor)).cpu().detach().numpy()
        print(probs.shape)
        print(probs[0])    
        probs = probs[0][len(prefix) - 1]
        del prefix_tensor
        return dict(zip(tokens, list(probs)))

def to_gpu(tensor, gpu):   
        
        if gpu > -1:
            return tensor.cuda(device=gpu)
        
        else:
            return tensor.cpu()
def compute_lengths(input_ix, eos_ix=token_to_id[EOS]):
    a = input_ix.eq(eos_ix)
    count_eos = torch.cumsum(a, 1)
    lengths = torch.sum(count_eos.eq(0), 1)
    return lengths + 1 

def sequence_mask(lengths, maxlen, dtype=torch.bool):
    """
    :param lenghts: array of size K, lenghts of input K lines
    :param maxlen: number of steps in our case
    """
    if maxlen is None:
        maxlen = lengths.max()
    cuda_check = lengths.is_cuda
    if cuda_check:
        cuda_device = lengths.get_device()
    
    one_tensor = torch.ones((len(lengths), maxlen))
    if (cuda_check):
        one_tensor = one_tensor.cuda(device=cuda_device)
    
    mask = ~(one_tensor.cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


def compute_loss(logits, targets):
    """
    :param model: language model that can compute next token logits given token indices
    :param input ix: int32 matrix of tokens, shape: [batch_size, length]; padded with eos_ix
    :return: scalar
    """
    lengths = compute_lengths(targets)
    logits = logits.permute(0, 2, 1)
    m = nn.LogSoftmax(dim=1)
    
    
    seq_m = sequence_mask(lengths=lengths, maxlen=m(logits).size()[2])
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    loss = criterion(m(logits), targets) 
    loss = seq_m * loss
        
    loss = torch.sum(loss, dim = 1)
    
    return torch.mean(loss)
def score_lines(dev_lines, batch_size, model):
    """ computes average loss over the entire dataset """
    dev_loss_num, dev_loss_len = 0., 0.
    for i in range(0, len(dev_lines), batch_size):
        batch_ix = to_matrix(dev_lines[i: i + batch_size])
        
        tg = to_gpu(torch.from_numpy(batch_ix[:, 1:]).to(torch.int64), model.gpu)
        
        input_ = to_gpu(torch.from_numpy(batch_ix[:, :-1]).to(torch.int64), model.gpu).to(torch.int64)
        
        loss_i = compute_loss(model(input_), tg)
        dev_loss_num += loss_i.cpu().detach().numpy() * len(batch_ix)
        dev_loss_len += len(batch_ix)
        del tg, input_, loss_i
    return dev_loss_num / dev_loss_len

def generate(lm, prefix=BOS, temperature=1.0, max_len=320):
    """
    Samples output sequence from probability distribution obtained by lm
    :param temperature: samples proportionally to lm probabilities ^ temperature
        if temperature == 0, always takes most likely token. Break ties arbitrarily.
    """
    while True:
        token_probs = lm.get_possible_next_tokens(prefix)
        tokens, probs = zip(*token_probs.items())
        if temperature == 0:
            next_token = tokens[np.argmax(probs)]
        else:
            probs = np.array([p ** (1. / temperature) for p in probs])
            probs /= sum(probs)
            next_token = np.random.choice(tokens, p=probs)
        
        prefix += next_token
        if next_token == EOS or len(prefix) > max_len: break
    return prefix

rnn_lm1 = RNNLanguageModel(gpu = 0)
rnn_lm1.cuda(device = rnn_lm1.gpu)  
"""
переделать код под мак(видеокарта), разобраться с кондой и pytorch
 """

from IPython.display import clear_output
from random import sample
from tqdm import trange, tnrange, tqdm_notebook


from sklearn.model_selection import train_test_split
train_lines, dev_lines = train_test_split(lines, test_size=0.2, 
                                          random_state=42)



optimizer = torch.optim.Adam(rnn_lm1.parameters(), lr=0.003)

batch_size = 256
score_dev_every = 250
train_history, dev_history = [], []

# dev_history.append((0, score_lines(dev_lines, batch_size, rnn_lm1)))

for i in tqdm_notebook(range(len(train_history), 15000)):
    batch = to_matrix(sample(train_lines, batch_size))  
    real_answer = to_gpu(torch.from_numpy(batch[:, 1:]).to(torch.int64), rnn_lm1.gpu)
    input_seq = to_gpu(torch.from_numpy(batch[:, :-1]).to(torch.int64), rnn_lm1.gpu).to(torch.int64)
    optimizer.zero_grad()
    logits = rnn_lm1(input_seq)
    loss_i = compute_loss(logits, real_answer)
    train_history.append((i, loss_i.detach().cpu().numpy()))
    
    loss_i.backward()
    optimizer.step()
    
    
    if (i + 1) % 50 == 0:
        clear_output(True)
        plt.scatter(*zip(*train_history), alpha=0.1, label='train_loss')
        if len(dev_history):
            plt.plot(*zip(*dev_history), color='red', label='dev_loss')
        plt.legend(); plt.grid(); plt.show()
        print("Generated examples (tau=0.5):")
        print (loss_i)
        for j in range(3):
            print(generate(rnn_lm1, temperature=0.5))
    
    if (i + 1) % score_dev_every == 0:
        print("Scoring dev...")
        dev_history.append((i, score_lines(dev_lines, batch_size, rnn_lm1)))
        print('#%i Dev loss: %.3f' % dev_history[-1])
    del logits, input_seq, loss_i

    for i in range(5):
        print(generate(rnn_lm1, temperature=0.25).strip())
