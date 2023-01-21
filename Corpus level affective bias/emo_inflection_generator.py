# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:38:21 2022

@author: Anoop
Emotion word inflection finder 

https://pypi.org/project/pyinflect/
https://www.merriam-webster.com/
"""

from pyinflect import getAllInflections


anger_terms = ['anger', 'irritability', 'exasperation', 'rage', 'disgust', 'envy', 
               'torment', 'aggravation', 'agitation', 'annoyance', 'grouchy', 'grumpy', 
               'crosspatch', 'frustration', 'outrage', 'fury', 'wrath', 'hostility', 
               'ferocity', 'bitterness', 'hatred', 'scorn', 'spite', 'vengefulness', 
               'dislike', 'resentment', 'revulsion', 'contempt', 'loathing', 'jealousy', 
               'angry', 'annoyed', 'enraged', 'furious', 'irritated', 'annoying', 
               'displeasing', 'irritating', 'outrageous', 'vexing', 'aggravated', 
               'outraged', 'wrathful', 'outraging', 'irritable', 'irritation','irritated',
               'irritates', 'irritating', 'irritability', 'irritations', 'exasperated', 
               'exasperating', 'raged', 'raging', 'rages', 'disgusts', 'distasteful', 
               'envies', 'tormented', 'aggravating', 'agitations', 'agitational', 
               'annoying', 'annoyingly', 'annoyed', 'annoying', 'annoys', 'annoy',
               'grouchier', 'grouchiest', 'grumpier', 'grumpiest', 'frustrating', 'frustrated', 
               'frustrates', 'outrages', 'outraged', 'outraging', 'furies', 'hostilities', 
               'hostile', 'ferocious', 'bitter', 'hatreds', 'spited', 'spiting', 'vengeful',
               'vengeance', 'vengefully', 'vengefulness', 'dislikes', 'revulsive', 'jealous',
               'jealousies', 'angrier', 'angriest', 'enrage', 'enraged', 'enraging', 'enrages', 
               'displease', 'displeased', 'displeasing', 'displeases', 
               'vexingly', 'wrathfully','wrathfulness','wrathfully']

fear_terms = ['fear', 'horror', 'nervousness', 'pity', 'sympathy', 'alarm', 'shock', 
              'fright', 'terror', 'panic', 'hysteria', 'mortification', 'anxiety', 'suspense', 
              'uneasiness', 'apprehension', 'worry', 'distress', 'dread', 'anxious', 
              'discouraged', 'fearful', 'scared', 'terrified', 'dreadful', 'horrible', 
              'shocking', 'terrifying', 'threatening', 'frightened', 'alarmed', 'panicked', 
              'alarming', 'forbidding', 'feared', 'fearing', 'fears', 'fearer', 
              'horrors', 'nervous', 'nervously', 'nervousness', 'pities', 'pitied', 
              'pitying', 'sympathies', 'shocked', 'shocking', 'shocks', 'shockable', 
              'frighted', 'frighting', 'frights', 'terrors', 'terrorless', 'panicked', 
              'panicking', 'anxieties', 'suspense', 'suspenseful', 'suspensefully', 
              'suspensefulness', 'suspenseless', 'uneasy', 'worried', 'worrying', 
              'worries', 'distressed', 'distressing', 'distresses', 'dreaded', 'dreading',
              'dreads', 'anxiously', 'anxiousness', 'discourage', 'discouraged', 
              'discouraging', 'fearfully', 'fearfulness', 'scare', 'scared', 'scaring', 
              'terrify', 'terrified', 'terrifying', 'dreadfully', 'dreadfulness', 
              'horribleness', 'horribly', 'threateningly', 'forbiddingly']

joy_terms = ['joy', 'cheerfulness', 'zest', 'contentment', 'pride', 'optimism', 'enthrallment', 
             'relief', 'amusement', 'bliss', 'gaiety', 'glee', 'jolliness', 'joviality', 'delight', 
             'enjoyment', 'gladness', 'happiness', 'jubilation', 'elation', 'satisfaction', 'ecstasy', 
             'euphoria', 'enthusiasm', 'zeal', 'excitement', 'thrill', 'exhilaration', 'pleasure', 
             'triumph', 'eagerness', 'hope', 'rapture', 'ecstatic', 'excited', 'glad', 'happy', 
             'relieved', 'amazing', 'funny', 'great', 'hilarious', 'wonderful', 'elated', 'delightful', 
             'pleasing', 'joyful', 'joyed', 'joying', 'joys', 'joyless', 'joylessly', 'joylessness', 
             'cheerful', 'cheer', 'cheerfully', 'cheerfulness',  'zestful', 'zestfully', 'zestfulness', 
             'zestless', 'prided', 'priding', 'enthralled', 'enthralling', 'amusing', 'amused', 
             'amusements', 'gaieties', 'gayety', 'jollily', 'jovial', 'delighter', 'enjoying', 'enjoy', 
             'enjoyed', 'enjoying', 'enjoys', 'enjoyable', 'enjoyableness', 'enjoyably', 'enjoyer', 
             'glad', 'gladder', 'gladdest', 'gladded', 'gladding', 'gladly', 'gladness', 'happy', 
             'happier', 'happiest', 'elated', 'satisfied', 'satisfy', 'satisfying', 'ecstasies', 
             'excites', 'exciting', 'excited', 'excite', 'thrilled', 'thrilling', 'thrills', 
             'exhilarating', 'exhilarated', 'pleasured', 'pleasuring', 'triumphs', 'triumphal',  
             'eager', 'eagerly', 'eagerness', 'hoped', 'hoping', 'raptured', 'rapturing', 'ecstasy', 
             'ecstasies', 'relievedly', 'funnier','funniest', 'delightfully', 'delightfulness']

sadness_terms = ['sadness', 'suffering', 'disappointment', 'shame', 'neglect', 'sympathy', 'agony', 
                 'anguish', 'hurt', 'depression', 'despair', 'gloom', 'glumness', 'unhappiness', 
                 'grief', 'sorrow', 'woe', 'misery', 'melancholy', 'dismay', 'displeasure', 
                 'guilt', 'regret', 'remorse', 'alienation', 'defeatism', 'dejection', 'embarrassment', 
                 'homesickness', 'humiliation', 'insecurity', 'insult', 'isolation', 'loneliness', 
                 'rejection', 'depressed', 'devastated', 'disappointed', 'miserable', 'sad', 
                 'depressing', 'gloomy', 'grim', 'heartbreaking', 'serious', 'melancholic', 'dejected', 
                 'saddening', 'sad', 'sadder', 'saddest', 'suffers', 'suffer', 'suffered', 
                 'disappointing', 'disappointed', 'disappoints', 'shamed', 'shaming', 'neglected', 
                 'neglecting', 'neglects', 'neglecter', 'sympathies', 'agonies', 'anguished', 
                 'anguishing', 'anguishes', 'hurting', 'depressions', 'despaired', 'despairing', 
                 'despairs', 'despairer', 'gloomed', 'glooming', 'glooms', 'glum', 'glummer', 
                 'glummest', 'glumly', 'glumness', 'unhappy', 'sorrowed', 'sorrowing', 'sorrows', 
                 'woes', 'miseries', 'melancholies', 'dismayed', 'dismaying', 'displeased', 'guilty', 
                 'guilted', 'guilting', 'guilts', 'regretted', 'regretting', 'embarrasses', 
                 'embarrassed', 'homesick', 'humiliate', 'humiliated', 'humiliating', 'insecurities', 
                 'insecure', 	'insulted', 'insulting', 'insults', 'insulter', 'isolating', 
                 'isolated', 'lonely', 'lonelier', 'loneliest', 'rejecting', 'rejected', 'depression', 
                 'devastated', 'miserable', 'miserableness', 'miserably', 'depresses', 'depression', 
                 'depressingly', 'gloomier', 'gloomiest', 'gloomily', 'gloominess', 'grimmer', 
                 'grimmest', 'grimly', 'grimness', 'seriousness', 'dejectedly', 'dejectedness', 
                 'saddened', 'sadden', 'saddening']

def emo_word_inflection_generator(word_list):
    emo_data = word_list
    lowercased_data = [x.lower() for x in emo_data]
    dupli_removed = list(dict.fromkeys(lowercased_data))
    for i in range(len(dupli_removed)):
        term = getAllInflections(dupli_removed[i])
        for val in term.values():
            emo_data.append(val[0])
            
    emo_data_final = list(dict.fromkeys(emo_data)) # removing duplication again 
    emo_data_final.sort()
    return emo_data_final



anger = emo_word_inflection_generator(anger_terms)
fear = emo_word_inflection_generator(fear_terms)    
joy = emo_word_inflection_generator(joy_terms)
sadness = emo_word_inflection_generator(sadness_terms)

