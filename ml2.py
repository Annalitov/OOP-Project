import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

pd.set_option('max_colwidth', 200)

df = pd.read_csv('shortjokes.csv', index_col=0, dtype=str)
print(df.tail(3))

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