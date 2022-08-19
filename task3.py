import numpy as np
from scipy.sparse.linalg import norm
from collections import Counter
import time 
from nltk.corpus import stopwords
import pandas as pd
import json
from task1 import generate_lemmas

from scipy.sparse import  lil_matrix

time_start = time.time() 

def generate_tfidf_scores(map_df,inverted_index):

    '''
    generates tfidf scores of query passage pairs. 

    Inputs:
    map_df (dataframe): Conatins passage and query pairs.
    columns contain qid,pid,queries,passages.

    inverted_index (dict): Inverted index dictionary

    Outputs:

    out (np.array): Array of cosine similarity scores 
    with each row containing qid,pid,scores 
    '''

    
    vocab = inverted_index.keys()
    n_vocab = len(vocab)

    vocab_list=  list(vocab)
    vocab_idx_list = list(range(n_vocab))

    # create word and index pairs
    vocab_dict = dict(zip(vocab_list,vocab_idx_list))  
    
    n = len(map_df)
    
    # row based list of list sparce matrix 
    tfidf_matrix_passage = lil_matrix((n,n_vocab))  
    tfidf_matrix_query = lil_matrix((n,n_vocab))

    for idx,map_row in enumerate(map_df.itertuples(index = False)):
        passage = map_row.passages
        query = map_row.queries 
        
        # tfidf vector for passages
        tf_dict_passage = Counter(passage)
        for word in tf_dict_passage:

            df = len(inverted_index[word])
            tf = 1 + np.log10(tf_dict_passage[word])     
            idf = np.log10(n_vocab/df)
            
            jdx = vocab_dict[word]
            tfidf_matrix_passage[idx,jdx] = tf*idf

        # tfidf vector for query  
        tf_dict_query = Counter(query)
        for word in tf_dict_query:
            if word in vocab:
                df = len(inverted_index[word])
                tf = 1 + np.log10(tf_dict_query[word]) 
                idf = np.log10(n_vocab/df)

                jdx = vocab_dict[word]
                tfidf_matrix_query[idx,jdx] = tf*idf
    
    # Convert to compressed sparse column matrix
    tfidf_matrix_passage = tfidf_matrix_passage.tocsc()
    tfidf_matrix_query = tfidf_matrix_query.tocsc()    
    
    # dot product of query and passage tfidf vectors
    a2 = tfidf_matrix_passage.multiply(tfidf_matrix_query)
    a3 = a2.sum(axis=1)
    
    # vector norm
    B = norm(tfidf_matrix_passage,axis = 1).reshape(-1,1)
    C = norm(tfidf_matrix_query,axis = 1).reshape(-1,1)

    D = np.multiply(B,C)

    # cosine similarity score
    score = a3/D
    score = score.reshape(-1,1)

    pid = map_df.pid
    qid = map_df.qid
    qid_arr = np.array(qid).reshape(-1,1)
    pid_arr = np.array(pid).reshape(-1,1)
    out = np.hstack((qid_arr,pid_arr,score))
     
    return out 

def generate_bm25_scores(map_df,inverted_index):
    '''
    generates bm25 scores of query passage pairs 

    Inputs:
    map_df (dataframe): Conatins passage and query pairs.
    columns contain qid,pid,queries,passages.

    inverted_index (dict): Inverted index dictionary

    Outputs:

    result (np.array): Array of bm25 scores 
    with each row containing qid,pid,scores 

    '''

    passage_len = map_df['passage_len'].to_numpy()
    avdl =np.sum(passage_len)/passage_len.shape[0] 
    #print(avdl)
    k1 = 1.2
    k2 = 100
    b = 0.75
    N = len(passage_df)
    result = []
    
    result = np.zeros((len(map_df),3))
    for idx, map_row in enumerate(map_df.itertuples(index=False)):
        qid = str(map_row.qid)
        query = map_row.queries
        pid = str(map_row.pid)
        passage = map_row.passages

        query_word_freq = Counter(query)
        
        dl = len(passage)
        K = k1*((1-b)+b*dl/avdl)
        score = 0
        for word in query:
            if word in inverted_index:
                ni  = len(inverted_index[word])
                if pid in inverted_index[word]:
                    fi  = inverted_index[word][pid]
                else:fi = 0
            else:  ni = 0
            
            qfi = query_word_freq[word]
            A = np.log((N-ni+0.5)/(ni+0.5))
            B = (k1 +1)*fi / (K+fi)
            C = (k2 +1)*qfi/(k2 +qfi)
            score += A*B*C

        result_i = np.array([qid,pid,score])

        result[idx] = result_i

    return result

def save_scores(scores,name='DEFAULT'):
    '''
    saves the model's score in a csv file

    inputs:
    scores : contains qid,pid and scores
    name (str): name of the csv five   
    
    '''
    scores_df = pd.DataFrame(scores,columns=['qid','pid','score'])
    scores_df = scores_df.sort_values(['score'],ascending=False).groupby('qid',sort = False).head(100).reset_index(drop=True)
    out_df = pd.merge(query_df,scores_df,on = 'qid',how='left')
    out_df['pid'] = out_df.pid.astype(int)
    out_df['qid'] = out_df.qid.astype(int)
    out_df = out_df[['qid','pid','score']]
    out_df.to_csv(f'{name}.csv',index=False,header=False)



if __name__ == "__main__":
    print("start!")

    # import inverted index
    with open("inverted_index.json",encoding = "utf8") as json_f:
        inverted_index= json.load(json_f)


    # read data
    time_start_a = time.time()
    df = pd.read_csv("candidate-passages-top1000.tsv",sep ='\t',header = None,names = ['qid','pid','queries','passages'])
    query_df = pd.read_csv("test-queries.tsv",sep ='\t',header = None,names = ['qid' ,'queries'])

    # drop duplicates
    map_df = df[['qid','pid']].drop_duplicates()
    query_df.drop_duplicates(inplace=True)
    passage_df = df[['pid','passages']].drop_duplicates()

    # remove stop words
    stopwords_list  = stopwords.words('english')
    stopwords_clean = set(generate_lemmas(str(stopwords_list)))
    query_series = query_df.queries
    queries_lemma =  [[query for query in generate_lemmas(queries) if query not in stopwords_clean] for queries in query_series]
    query_df['queries'] = queries_lemma
  
    
    # clean data
    series_passage = passage_df.passages 
    passage_lemma =  [[lemma for lemma in generate_lemmas(passage) if lemma not in stopwords_clean] for passage in series_passage]
    passage_df['passages'] = passage_lemma
    passage_df['passage_len'] = passage_df.passages.apply(len) # for bm25
    
    # merge clean data to the mapping
    map_df = map_df.merge(query_df,how = 'left',on = 'qid')
    map_df = map_df.merge(passage_df,how = 'left',on = 'pid')

    # tfidf
    scores = generate_tfidf_scores(map_df,inverted_index)
    save_scores(scores,name ='tfidf')

    #bm25
    scores = generate_bm25_scores(map_df,inverted_index)
    save_scores(scores,name ='bm25')
       
    time_end = time.time()
    print(f"Total time of execution : {(time_end-time_start)/60} mins")

           

