#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:48:30 2022

@author: user
https://github.com/tanyichern/social-biases-contextualized
"""
from collections import Counter
from multiprocessing import Pool
import datetime
import sys

# global variables 
global m_pns
global f_pns
global n_pns
global m_tgts
global f_tgts
    
def update_count_para(sentence):
    counts = {"male_pronouns": 0,
              "female_pronouns": 0,
              "neutral_pronouns": 0,
              "male_pro_stereo": 0,
              "male_anti_stereo": 0,
              "female_pro_stereo": 0,
              "female_anti_stereo": 0,
              "male_neutral": 0,
              "female_neutral": 0}
    counts = Counter(counts)
    tokens = sentence.lower().split()
    male_occ = False
    female_occ = False
    neutral_occ = False

    for m_pn in m_pns:
        if m_pn in tokens:
            male_occ = True
            counts["male_pronouns"] += 1
            break

    for f_pn in f_pns:
        if f_pn in tokens:
            female_occ = True
            counts["female_pronouns"] += 1
            break

    for n_pn in n_pns:
        if n_pn in tokens:
            neutral_occ = True
            counts["neutral_pronouns"] += 1
            break

    for m_tgt in m_tgts:
        if m_tgt in tokens and male_occ:
            counts["male_pro_stereo"] += 1
        if m_tgt in tokens and female_occ:
            counts["female_anti_stereo"] += 1
        if m_tgt in tokens and neutral_occ:
            counts["male_neutral"] += 1


    for f_tgt in f_tgts:
        if f_tgt in tokens and female_occ:
            counts["female_pro_stereo"] += 1
        if f_tgt in tokens and male_occ:
            counts["male_anti_stereo"] += 1
        if f_tgt in tokens and neutral_occ:
            counts["female_neutral"] += 1


    return counts

sentence = ["her she developer was his CEO , but his designer and her CEO was hers.",
            "her developer was he ceo",
            "he they designer ceo"]

m_tgts = ["developer", "ceo"]
f_tgts = ["designer"]

m_pns = ["he", "him", "his"]
f_pns = ["she", "her", "hers"]
n_pns = ["they", "them", "their", "theirs"]



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

