#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding = 'utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    with open(filepath,'r', encoding = 'utf-8')as doc:
        for word in doc:
            word = word.strip()  
            if word in vocab:
                if not word in bow.keys():
                    bow[word] = 1
                else:
                    bow[word] += 1
        
            if not word in vocab:
                if None in bow.keys():
                    bow[None] += 1
                else:
                    bow[None] = 1

    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    freq = {}
            
    for file in training_data:
        if file["label"] in label_list:
            if file["label"] in freq:
                freq[file["label"]] += 1
            else:
                freq[file["label"]] = 1
    
    for label in label_list:
        logprob[label] = math.log((freq[label] + smooth) / (len(training_data) + smooth * len(label_list)))

    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    
    counts = {}
    for word in vocab:
        counts[word] = 0
    counts[None] = 0
    
    total = 0
    for doc in training_data:
        if doc["label"] == label:
            for key in doc['bow'].keys():
                if key in counts.keys():
                    counts[key] += doc['bow'][key]
                else:
                    counts[None] += counts['bow'][key]
                    
                total += doc['bow'][key]
        
    for word in counts:
        word_prob[word] = math.log((counts[word] + smooth*1) / (total + smooth * (len(vocab) + 1)))
     
    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    prior_log = prior(training_data, label_list)
    p_word_2020 = p_word_given_label(vocab, training_data, '2020')
    p_word_2016 = p_word_given_label(vocab, training_data, '2016')
    
    retval['vocabulary'] = vocab
    retval['log prior'] = prior_log
    retval['log p(w|y=2020)'] = p_word_2020
    retval['log p(w|y=2016)'] = p_word_2016

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    doc_2020 = model['log prior']['2020']
    doc_2016 = model['log prior']['2016']
    
    with open(filepath, 'r', encoding='utf-8') as doc:
        for word in doc:
            word = word.strip()
            if word in model['vocabulary']:
                doc_2020 += model['log p(w|y=2020)'][word]
                doc_2016 += model['log p(w|y=2016)'][word]
            else:
                doc_2020 += model['log p(w|y=2020)'][None]
                doc_2016 += model['log p(w|y=2016)'][None]

    retval['log p(y=2020|x)'] = doc_2020
    retval['log p(y=2016|x)'] = doc_2016
    if doc_2020 > doc_2016:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
        
    return retval
                






