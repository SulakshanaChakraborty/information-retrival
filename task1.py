import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import re 
from collections import Counter
import time 
import matplotlib.pyplot as plt
import json
import pandas as pd

time_start = time.time()
######### pre-processing
def generate_lemmas(content):
    '''
    Pre-process input and generate lemmas

    input: 
    content(str): Text to lemmatize

    output:
    lemma_list (list): list of lemmas 
    '''

    def remove_special_characters(passage):
        '''
        replaces special charcters with whitespaces
        '''

        return re.sub(r'([^a-zA-Z\s]+?)',' ',passage)

    def standerdise_text(passage):
        '''
        Performs multiple standerdisation steps on the input string
        1. standardises abbreviations of country names
        2. removes apostrophe in contractions
        3. removes hyphens from commonly hyphenated words
        '''

        # map country names
        rx=r'\b(?:{})\b'.format("|".join(['u.s.a','u.s','usa']))
        passage = re.sub(rx,'us',passage)
        passage = re.sub(r'\bu.k\b','uk',passage)

        #remove contractions    
        rx=r"([A-Za-z]+)[\'`]\b({})\b".format("|".join(["t","ll","ve","re","m","d","s"]))   
        passage = re.sub(rx, r'\1'r'\2', passage)

        # group common hypenated words
        
        # prefix
        rx=r'\b({})\b[\-]([A-Za-z]+)'.format("|".join(['co','post','de','non','re','anti','pro','bi','multi','semi','un','sub']))   
        passage = re.sub(rx, r'\1'r'\2', passage)
        # suffix
        rx=r"([A-Za-z]+)[\-]\b({})\b".format("|".join(["off","in","offs","less","like"])) 
        passage = re.sub(rx, r'\1'r'\2', passage)

        return passage
    

    content = content.lower()
    content = standerdise_text(content)
    preprocessed_txt = remove_special_characters(content)

    tokens = nltk.word_tokenize(preprocessed_txt) 
   
    lemmatizer = WordNetLemmatizer()        
    lemma_list = [lemmatizer.lemmatize(word) for word in tokens]

    return lemma_list


######### Zipf's law

def calculate_word_prob(corpus):

    '''
    Calculates the zipf's theoretical probabilities and actual probabilities of the word and
    generates graphs to compare the deviation from theoretical vs actual probailities.
    Saves the comparison garphs in pdf files

    inputs:
    corpus (list): list of tokens of a corpus

    '''

    dict_freq = Counter(corpus)

    # normalized prob
    constant = sum(dict_freq.values())
    freq_array = np.array(list(dict_freq.values()))
    word_prob =  (-np.sort(-freq_array))/ constant
    number_of_elem = len(dict_freq)
    rank = np.arange(1,number_of_elem+1)


    # zips law
    arr_const = np.hstack((rank[:20,None],word_prob[:20,None]))
    df_const = pd.DataFrame(data = arr_const)
    df_const.to_csv('Zipfs_law_comp_top20.csv',index=False,header=False)

    # plot log scaled graph
    norm_const = sum([1/x for x in range(1,number_of_elem+1)])
    zip_f = 1/(rank*norm_const)
    plt.loglog(rank,zip_f,label = "theory (Zipf's law)")
    plt.loglog(rank,word_prob,label = "data")
    plt.xlabel('Term frequency ranking (log)')
    plt.ylabel('Term prob. of occurrence (log)')
    plt.legend()
    plt.savefig('plot_ziph_loglog.pdf')
    plt.close()

    # plot graph
    plt.plot(rank,word_prob,label = "data")
    plt.plot(rank,zip_f,label = "theory (Zipf's law)")
    plt.xlabel('Term frequency ranking ')
    plt.ylabel('Term prob. of occurrence')
    plt.legend()
    plt.savefig('plot_ziph.pdf')
    plt.close()


    return dict_freq

if __name__ == "__main__":

    print("start!")

    with open("passage-collection.txt",encoding="utf8") as f:
        content = f.read()

    # pre-process and lemmatise
    lemma_list = generate_lemmas(content)
  
    # zipf's law vs actual freq
    dict_vocab_freq = calculate_word_prob(lemma_list)
    print("vocab len",len(dict_vocab_freq.keys()))
 
    with open("vocab_freq.json",encoding="utf8",mode = 'w') as json_f:
        json.dump(dict_vocab_freq,json_f)

    time_end = time.time()
    print("total time of execution",(time_end-time_start)/60)
    




