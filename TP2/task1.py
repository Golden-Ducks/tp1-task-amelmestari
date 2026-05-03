import re 
import numpy as np
from sklearn.cluster import KMeans

doc1="The gold medal price is high effort"
doc2= "Winning a gold medal needs a high jump"
doc3="Market for a gold medal is a trade of sweat"
doc4="The athlete will trade all for a gold medal"
doc5="The gold bars price is high today"
doc6="Investing in gold bars needs a high rate"
doc7="Market for gold bars is a trade of money"
doc8="The bank will trade all for gold bars"

all_docs=[doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8]

def preprocess_text(text):
    text = text.lower()
    text=re.sub(r'[^\w\s]', '', text)
    tokens=text.split()
    return tokens

def generate_ngrams(tokens,n):

    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def vectorize(doc,n_gram_size=1):
    preprocessed_docs=[]
    all_ngrans=[]

    for i in doc:
        tokens=preprocess_text(i)
        grams=generate_ngrams(tokens,n_gram_size)
        preprocessed_docs.append(grams)
        all_ngrans.extend(grams)
    
    vocab=sorted(list(set(all_ngrans)))
    vocab_index={word:i for i,word in enumerate(vocab)}
    x=np.zeros((len(doc),len(vocab)))

    for i,j in enumerate(preprocessed_docs):
        for gram in j:
            a=vocab_index[gram]
            x[i,a]+=1
    return x

x1=vectorize(all_docs,1)
km1=KMeans(n_clusters=2,random_state=42).fit(x1)

print ("1-gram clusters",km1.labels_)

x2=vectorize(all_docs,2)
km2=KMeans(n_clusters=2,random_state=42).fit(x2)

print ("2-gram clusters",km2.labels_)