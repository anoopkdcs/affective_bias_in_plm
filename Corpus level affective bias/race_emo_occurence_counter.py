#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:08:31 2022

@author: Anoop K
https://github.com/tanyichern/social-biases-contextualized
Corpus level affective bias in Race: co-occurrence of race emotion terms 
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
global afriame_tgts
global euroame_tgts

    



def update_count_para(sentence):
    counts = {"anger_count": 0,
              "fear_count": 0,
              "joy_count": 0,
              "sadness_count": 0,
              "anger_afri_amer": 0,
              "fear_afri_amer": 0,
              "joy_afri_amer": 0,
              "sadness_afri_amer": 0,
              "anger_euro_amer": 0,
              "fear_euro_amer": 0,
              "joy_euro_amer": 0,
              "sadness_euro_amer":0
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

#for african american 
    for afriame_tgt in afriame_tgts:
        if afriame_tgt in tokens and anger_occ:
            counts["anger_afri_amer"] += 1
            break
        if afriame_tgt in tokens and fear_occ:
            counts["fear_afri_amer"] += 1
            break
        if afriame_tgt in tokens and joy_occ:
            counts["joy_afri_amer"] += 1
            break
        if afriame_tgt in tokens and sadness_occ:
            counts["sadness_afri_amer"] +=1
            break

#for european american 
    for euroame_tgt in euroame_tgts:
        if euroame_tgt in tokens and anger_occ:
            counts["anger_euro_amer"] += 1
            break
        if euroame_tgt in tokens and fear_occ:
            counts["fear_euro_amer"] += 1
            break
        if euroame_tgt in tokens and joy_occ:
            counts["joy_euro_amer"] += 1
            break
        if euroame_tgt in tokens and sadness_occ:
            counts["sadness_euro_amer"] += 1
            break

    return counts

sentence = ["he him anger irritability",
            "she sadness",
            "his joy",
            "he is anger about homosexual "]

#Race target words 
afriame_tgts = ["he", "him", "his"] #african american 
euroame_tgts = ["she", "her", "hers"] #european  american

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