#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:07:00 2022

@author: Anoop K
https://github.com/tanyichern/social-biases-contextualized
Corpus level affective bias in Religion: co-occurrence of race emotion terms 
"""

from collections import Counter
from multiprocessing import Pool
import datetime
import sys

# global variables 
global anger_wdrs
global fear_wrds
global joy_wrds
global sadness_wrds
global muslim_tgts
global jewish_tgts
global christian_tgts

    



def update_count_para(sentence):
    counts = {"anger_count": 0,
              "fear_count": 0,
              "joy_count": 0,
              "sadness_count": 0,
              "anger_muslim": 0,
              "fear_muslim": 0,
              "joy_muslim": 0,
              "sadness_muslim": 0,
              "anger_jewish": 0,
              "fear_jewish": 0,
              "joy_jewish": 0,
              "sadness_jewish":0,
              "anger_christian": 0,
              "fear_christian": 0,
              "joy_christian": 0,
              "sadness_christian":0
              }
    
    counts = Counter(counts)
    tokens = sentence.lower().split()
    anger_occ = False
    fear_occ = False
    joy_occ = False
    sadness_occ = False

    for anger_wdr in anger_wdrs:
        if anger_wdr in tokens:
            anger_occ = True
            counts["anger_count"] += 1
            break

    for fear_wrd in fear_wrds:
        if fear_wrd in tokens:
            fear_occ = True
            counts["fear_count"] += 1
            break

    for joy_wrd in joy_wrds:
        if joy_wrd in tokens:
            joy_occ = True
            counts["joy_count"] += 1
            break

    for sadness_wrd in sadness_wrds:
        if sadness_wrd in tokens:
            sadness_occ = True
            counts["sadness_count"] += 1
            break

#for muslim
    for muslim_tgt in muslim_tgts:
        if muslim_tgt in tokens and anger_occ:
            counts["anger_muslim"] += 1
            break
        if muslim_tgt in tokens and fear_occ:
            counts["fear_muslim"] += 1
            break
        if muslim_tgt in tokens and joy_occ:
            counts["joy_muslim"] += 1
            break
        if muslim_tgt in tokens and sadness_occ:
            counts["sadness_muslim"] +=1
            break

#for jewish
    for jewish_tgt in jewish_tgts:
        if jewish_tgt in tokens and anger_occ:
            counts["anger_jewish"] += 1
            break
        if jewish_tgt in tokens and fear_occ:
            counts["fear_jewish"] += 1
            break
        if jewish_tgt in tokens and joy_occ:
            counts["joy_jewish"] += 1
            break
        if jewish_tgt in tokens and sadness_occ:
            counts["sadness_jewish"] += 1
            break

#for christian
    for christian_tgt in christian_tgts:
        if christian_tgt in tokens and anger_occ:
            counts["anger_christian"] += 1
            break
        if christian_tgt in tokens and fear_occ:
            counts["fear_christian"] += 1
            break
        if christian_tgt in tokens and joy_occ:
            counts["joy_christian"] += 1
            break
        if christian_tgt in tokens and sadness_occ:
            counts["sadness_christian"] += 1
            break

    return counts

sentence = ["he him anger irritability",
            "she sadness",
            "his joy cristian",
            "he is anger about homosexual "]

#Religion target words 
muslim_tgts = ["he", "him", "his"] #muslim 
jewish_tgts = ["she", "her", "hers"] #jewish
christian_tgts = ["cristian"] #christian

anger_wdrs = ["anger","irritability"]
fear_wrds = ["fear", "horror"]
joy_wrds = ["joy", "cheerfulness"]
sadness_wrds = ["sadness","suffering"]



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