
from collections import Counter
import time 
from nltk.corpus import stopwords
import pandas as pd
import json
from task1 import generate_lemmas

time_start = time.time() 

def generate_inverted_index(vocab,passage_df):
    '''
    generates inverted index of the passage passed 
    the inverted index is stored in a dictionary,
    in the following structure:

    {"word 1" : {"pid 1" : count , "pid 2" : count}
     "word 2" : {"pid 4" : count , "pid 7" : count}}

    where,
    pid -> passage id 
    count -> frequency of the word occurance in the passage 

    inputs:
    vocab (set): vocabulary of the corpus
    passage_df (dataframe): contains passages and the corresponding pid

    outputs:
    inverted_index (dict): The inverted index

    '''   
    inverted_index ={}
    for vocab_word in vocab:
        inverted_index[vocab_word] = {}
        
    for passage_row in passage_df.itertuples(index = False):
                passage = passage_row.passage
                pid = passage_row.pid
                tf_dict = Counter(passage)
                for word in tf_dict:
                        inverted_index[word][pid] =  tf_dict[word]

    return inverted_index


if __name__ == "__main__":

    with open("vocab_freq.json",encoding = "utf8") as json_f:
        vocab = json.load(json_f)
    
    stopwords_list  = stopwords.words('english')
    stopwords_clean = set(generate_lemmas(str(stopwords_list)))

    # remove stop words from vocab
    for stopword in stopwords_clean: 
        try:
            del vocab[stopword]
        except: pass   

    # unique vocab     
    vocab =  set(vocab.keys())

    passage_df = pd.read_csv("candidate-passages-top1000.tsv",sep ='\t',\
    header = None,usecols = [1,3],names = ['pid','passage'])
  
    passage_df.drop_duplicates(inplace = True)
    series_passage = passage_df.passage 

    # pre-process passages 
    passage_lemma =  [[lemma for lemma in generate_lemmas(passage) if lemma not in stopwords_clean] for passage in series_passage]
    passage_df['passage'] = passage_lemma

    passage_dict = {}
    passage_dict['text'] = passage_lemma
    time_end_a = time.time()
    with open('passage_lemmas.json',encoding="utf8",mode = "w") as f:
            json.dump(passage_dict,f)
    
    # generate inverted index
    time_start_b = time.time()
    inverted_index = generate_inverted_index(vocab,passage_df)
    with open('inverted_index.json',encoding="utf8",mode = "w") as f:
            json.dump(inverted_index,f)

    time_end = time.time()
    print(f"total time of execution: {(time_end-time_start)/60} mins")