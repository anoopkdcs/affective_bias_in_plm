# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 20:34:45 2022

@author: Anoop K
 find the occurence of non-binary gender terms in SemEval-2018
"""
from collections import Counter
import multiprocessing as mp
from multiprocessing import Pool
import datetime
import sys
from numba import jit, cuda
import pickle
import numpy as np 

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string
import re
import io
from nltk.tokenize import sent_tokenize
from collections import Counter


# global variables 
global non_binary_wrds

def pkl_reader (file_path):
    with open(file_path, 'rb') as f:
        pkl_file = pickle.load(f)
        f.close()
        del f
    return pkl_file


def pre_processing(data):
  Tokens = []
  finalTokens =[]
  tokenizer = RegexpTokenizer(r'\w+')
  stop_words = set(stopwords.words('english')) 
  for i in range(len(data)):
    tempTokens = data[i].lower() #converting to lower case
    tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;1234567890"))
    tempTokens = tokenizer.tokenize(tempTokens) #tokenization 
    finalTokens.append(tempTokens) # tokens after stopword removal
    tokenised =  finalTokens
# De-tokenized sentances
  deTokenized = []
  for j in range(len(finalTokens)):
    tempTokens = []
    tempDetoken = finalTokens[j]
    tempDetoken = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tempDetoken]).strip()
    deTokenized.append(tempDetoken)

  return deTokenized


def update_count_para(sentence):
    counts = {"non_binary_count": 0,}
    
    counts = Counter(counts)
    tokens = sentence.lower().split()
    non_binary_occ = False

    for non_binary_wrd in non_binary_wrds:
        if non_binary_wrd in tokens:
            non_binary_occ = True
            print(non_binary_wrd)
            counts["non_binary_count"] += 1
            break
    return counts


#read SemEVal
train_data = np.load('data/semeval_2018/train_data_fused.npy')
test_data = np.load('data/semeval_2018/test_data.npy')

train_data_final = pre_processing(train_data)
test_data_final = pre_processing(test_data)

sentence = np.append(train_data_final,test_data_final)
non_binary_wrds = pkl_reader('target_terms/gender_terms/non-binary.pkl')



print("Number of processors: ", mp.cpu_count())
print(f"starting count, {datetime.datetime.now()}")
sys.stdout.flush()
workers = 16

with Pool(processes=workers) as pool:
    results = pool.map(update_count_para, sentence)

print(f"finished count, {datetime.datetime.now()}")
sys.stdout.flush()

# accumulate counts
print(f"starting acc, {datetime.datetime.now()}")
sys.stdout.flush()
master_count = sum(results, Counter())
print(f"finished acc, {datetime.datetime.now()}")
sys.stdout.flush()
    
# print counts
for key, value in master_count.items():
    print(f"{key}: {value}")
    
'''
output 
non_binary_count: 2
castrated
sissified
'''

