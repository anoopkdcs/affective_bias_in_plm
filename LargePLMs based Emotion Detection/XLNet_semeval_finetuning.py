#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import transformers
import keras 

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


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
plt.style.use('seaborn')

from transformers import TFXLNetModel, XLNetTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

print("TF = ", tf.__version__)
print("Keras = ", keras.__version__ )
print("Ttansformer = ", transformers.__version__)
print("NLTK = ", nltk.__version__)
print(tf.config.list_physical_devices('GPU'))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

######################## read data ################################
train_data = np.load('fine_tuning_data/semeval_2018/train_data_fused.npy')
train_labels = np.load('fine_tuning_data/semeval_2018/train_labels_fused.npy')

test_data = np.load('fine_tuning_data/semeval_2018/test_data.npy')
test_labels = np.load('fine_tuning_data/semeval_2018/test_labels.npy')

print("--------------Data Shapes-----------")
print("Train data: ", train_data.shape)
print("Train labels: ", train_labels.shape)
print("\n")

print("Test data: ", test_data.shape)
print("Test labels: ", test_labels.shape)

def pre_processing(data):
  Tokens = []
  finalTokens =[]
  tokenizer = RegexpTokenizer(r'\w+')
  stop_words = set(stopwords.words('english')) 
  for i in range(len(data)):
    tempTokens = data[i].lower() #converting to lower case
    tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;1234567890"))
    tempTokens = tokenizer.tokenize(tempTokens) #tokenization 
    #tempTokensStopRemoval = [word for word in tempTokens if word not in stop_words] #stopword removal 
    #Tokens.append(tempTokens) # tokens with out stopword removal 
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

########################## Data pre-processing #######################
train_data_final = pre_processing(train_data)
test_data_final = pre_processing(test_data)

########################## Label pre-processing #######################
train_labels_final = to_categorical(train_labels,num_classes=4)
test_labels_final = to_categorical(test_labels,num_classes=4)

print("train label shape: ", train_labels_final.shape)
print("test label shape: ", test_labels_final.shape)

########################## Dataset Statistics Extractor #######################
### Sent Conter
def sent_counter(data):
    sent = np.zeros((0,1))
    for i in range(len(data)):
        doc = data[i]
        number_of_sentences = sent_tokenize(doc)
        sent = np.append(sent,len(number_of_sentences))
        
    avg_sent = np.sum(sent)/len(data)
    return avg_sent


### word Conter (avg words and unique words)
def word_count(data):
    Tokens = []
    totalLength = 0
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(data)):
        tempTokens = data[i].lower() #converting to lower case
        tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;"))
        tempTokens = tokenizer.tokenize(tempTokens)
        Tokens.append(tempTokens)
        totalLength = totalLength + len(Tokens[i])
    AvgWordperDocument = totalLength/len(data)
    
    #Unique number of words 
    totalWordlist = []
    for j in range(len(Tokens)):
        totalWordlist.extend(Tokens[j])   
    wrd_counter = Counter(totalWordlist)
        
    return AvgWordperDocument,len(wrd_counter)

#Sentence Counter
train_sent = sent_counter(train_data_final)
print("Train data: ", train_sent) 

test_sent = sent_counter(test_data_final)
print("Test data: ", test_sent)

#Word Counter 
train_avg_word, train_uniqu_words = word_count(train_data_final)
print("Avg word in train data: ", train_avg_word) 
print("Unique words in train data: ", train_uniqu_words) 
print("\n")


test_avg_word, test_unique_words = word_count(test_data_final)
print("Avg word in test data: ", test_avg_word) 
print("Unique words in test data: ", test_unique_words) 

def get_inputs(tweets, tokenizer, max_len=30):
    """ Gets tensors from text using the tokenizer provided"""
    inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in tweets]
    inp_tok = np.array([a['input_ids'] for a in inps])
    ids = np.array([a['attention_mask'] for a in inps])
    segments = np.array([a['token_type_ids'] for a in inps])
    return inp_tok, ids, segments

# This is the identifier of the model. The library need this ID to download the weights and initialize the architecture
# here is all the supported ones:
# https://huggingface.co/transformers/pretrained_models.html
xlnet_model = 'xlnet-large-cased' #xlnet-base-cased-spiece.model, xlnet-large-cased
xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
x_train_tokens, x_train_ids, x_train_segments = get_inputs(train_data_final, xlnet_tokenizer)
x_test_tokens, x_test_ids, x_test_segments = get_inputs(test_data_final, xlnet_tokenizer)


print("train token shape:" +str(x_train_tokens.shape))
print("test token shape:" +str(x_test_tokens.shape))

print(x_train_tokens[1])
print(x_test_tokens[0])


def create_xlnet(mname):
    """ Creates the model. It is composed of the XLNet main block and then
    a classification head its added
    """
    # Define token ids as inputs
    word_inputs = tf.keras.Input(shape=(30,), name='word_inputs', dtype='int32')

    # Call XLNet model
    xlnet = TFXLNetModel.from_pretrained(mname)
    xlnet_encodings = xlnet(word_inputs)[0]

    # CLASSIFICATION HEAD 
    # Collect last step from last hidden state (CLS)
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    # Apply dropout for regularization
    #doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)
    # Final output 
    outputs = tf.keras.layers.Dense(4, activation='softmax', name='outputs')(doc_encoding)

    # Compile model
    model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

xlnet = create_xlnet(xlnet_model)

xlnet.summary()


checkpointer = [tf.keras.callbacks.ModelCheckpoint(filepath='models/xlnet/xlnet_semeval_lr000001_bs80.h5', verbose=1, 
                                                   save_best_only=True, save_weights_only=True,
                                                   monitor='val_accuracy', mode='max')
                #, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.02, 
                #restore_best_weights=True),
                #tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0)
                ]


history = xlnet.fit(x=x_train_tokens, y=train_labels_final, 
                    validation_data=(x_test_tokens,test_labels_final), 
                    epochs=50, batch_size=80, shuffle=True, callbacks=[checkpointer]) 

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


best_model = create_xlnet(xlnet_model)
best_model.load_weights('models/xlnet/xlnet_semeval_lr000001_bs80.h5')

predict_test = best_model.predict(x_test_tokens)
y_predicted = np.argmax(predict_test, axis = 1)


predict_test[10:15]

class_names=['anger', 'fear', 'joy', 'sadness']
print(classification_report(np.int32(test_labels), y_predicted,target_names=class_names))

cm = confusion_matrix(y_target=np.int32(test_labels), y_predicted=np.reshape(y_predicted,(len(y_predicted),1)), binary=False)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_normed=True,
                                cmap="YlGnBu",
                                colorbar=True,
                                class_names=class_names)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names,rotation=0)
plt.yticks(tick_marks, class_names)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')


# https://www.analyticsvidhya.com/blog/2021/12/multiclass-classification-using-transformers/
# 
# https://www.kaggle.com/harshpraharaj98/text-classification-using-bert-and-xlnet
# 
# https://www.kaggle.com/alvaroibrain/xlnet-huggingface-transformers
