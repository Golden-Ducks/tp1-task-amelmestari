import numpy as np
DO1= "I love cats"
Do2="Cats are chill"
DO3= "I am late"

docs=[DO1,Do2,DO3]

def preprocessing(text):
    return text.lower().split()
def add_padding(tokens):
     return ["<s>"] + tokens + ["</s>"]

def extract_windows(tokens,window_size=1):
     windows=[]
     for i in range(window_size,len(tokens)-window_size):
          window=tokens[i-window_size:i+window_size+1]
          windows.append("".join(window))
     return windows
def build_vocab(all_windows):
     vocab=sorted(list(set(all_windows)))
     return{w:i for i,w in enumerate(vocab)}

def vectorize_doc(doc_windows,vocab):
     vec=np.zeros(len(vocab))
     for w in doc_windows:
          if w in vocab:
               vec[vocab[w]]=1
     return vec
all_windows=[]
docs_windows=[]
for doc in docs:
     tokens=preprocessing(doc)
     tokens=add_padding(tokens)
     windows=extract_windows(tokens)
     docs_windows.append(windows)
     all_windows.extend(windows)
vocab=build_vocab(all_windows)
vectors=[]
for windows in docs_windows:
     vec=vectorize_doc(windows,vocab)

     vectors.append(vec)

X=np.array(vectors)

print("Vocabulary:\n",vocab)
print("Vectors:\n")
print(X)