import numpy as np
import time 
from nltk.corpus import stopwords
import pandas as pd
import json
from task1 import generate_lemmas

time_start = time.time()

def laplace_smoothing(map_df,inverted_index):

    '''
    Laplace smoothing negative likelihood scores of query passage pairs. 

    Inputs:
    map_df (dataframe): Conatins passage and query pairs.
    columns contain qid,pid,queries,passages.

    inverted_index (dict): Inverted index dictionary

    Outputs:

    result (np.array): Array of neg likelihood scores 
    with each row containing qid,pid,scores 
    '''

    result = np.zeros((len(map_df),3))

    for idx,map_row in enumerate(map_df.itertuples(index = False)):
        qid = str(map_row.qid)
        query = map_row.queries
        pid = str(map_row.pid)
        passage = map_row.passages
        D = len(passage)
        V = len(inverted_index.keys())
        
        score = 0
        
        for word in set(query):
            if word in inverted_index:
                if pid in inverted_index[word]:
                    mi = inverted_index[word][pid]
                    score += np.log((mi+1)/(D + V))
                else:
                    score+=np.log(1/(D+V))
        
        result[idx] = np.array([qid,pid,score])
    return result

def lidstone_smoothing(map_df,inverted_index):
    '''
    Lidstone smoothing negative likelihood scores of query passage pairs. 

    Inputs:
    map_df (dataframe): Conatins passage and query pairs.
    columns contain qid,pid,queries,passages.

    inverted_index (dict): Inverted index dictionary

    Outputs:

    result (np.array): Array of neg likelihood scores 
    with each row containing qid,pid,scores 

    '''
    
    result = np.zeros((len(map_df),3))
   
    for idx,map_row in enumerate(map_df.itertuples(index = False)):
        qid = str(map_row.qid)
        query = map_row.queries
        pid = str(map_row.pid)
        passage = map_row.passages
        D = len(passage)
        V = len(vocab)
        e = 0.1
        score = 0
        for word in set(query):
            if word in inverted_index:
                if pid in inverted_index[word]:
                    mi = inverted_index[word][pid]
                    score += np.log((mi+e)/(D + e*V))
                else:
                    score +=np.log(e/(D+e*V))

        result[idx] = np.array([qid,pid,score])
    
    return result

def dirichlet_smoothing(map_df,inverted_index):
    '''
    Dirichlet smoothing negative likelihood scores of query passage pairs. 

    Inputs:
    map_df (dataframe): Conatins passage and query pairs.
    columns contain qid,pid,queries,passages.

    inverted_index (dict): Inverted index dictionary

    Outputs:

    result (np.array): Array of neg likelihood scores 
    with each row containing qid,pid,scores 

    '''

    # count number of words in collection 
    with open("passage_lemmas.json",encoding = "utf8") as json_f:
        passage_lemmas= json.load(json_f)
    C=0
    for passage in passage_lemmas['text']:
         for word in passage:
             C+=1

    result = np.zeros((len(map_df),3))
    mu = 50
    for idx,map_row in enumerate(map_df.itertuples(index = False)):
        qid = str(map_row.qid)
        query = map_row.queries
        pid = str(map_row.pid)
        passage = map_row.passages

        D = len(passage)
        lmbd = D/(D+mu)
        q_set = set(query)

        # initialize to 0
        #x1,x2,score,fqi,cqi = 0,0,0,0,0

        score = 0
        for word in q_set:
            x1 = 0
            x2 = 0
            if word in inverted_index:
                cqi =  sum(inverted_index[word].values()) 
                x1 = ((1-lmbd)*cqi/C)
                if pid in inverted_index[word]:
                 fqi = inverted_index[word][pid]
                 x2 = (lmbd*fqi/D) 
            
            score_i = x1+x2
            if score_i > 0:
                score += np.log(score_i)

        result[idx] = np.array([qid,pid,score])
    
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
    
    ## import inverted index
    with open("inverted_index.json",encoding = "utf8") as json_f:
        inverted_index= json.load(json_f)

    vocab = set(inverted_index.keys())

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
    
    # laplace smoothing
    result_laplace = laplace_smoothing(map_df,inverted_index)
    save_scores(result_laplace,name = 'laplace')

    # lidstone smoothing
    result_lidstone = lidstone_smoothing(map_df,inverted_index)
    save_scores(result_lidstone,name = 'lidstone')
    
    # dirichlet smoothing
    result_dirichlet = dirichlet_smoothing(map_df,inverted_index)
    save_scores(result_dirichlet,name = 'dirichlet')

    time_end = time.time()
    print(f"total time of execution: {(time_end-time_start)/60} mins")

