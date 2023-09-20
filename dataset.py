# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:35:49 2022

@author: Xxp
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

torch.manual_seed(50)

# from the csv file extract train, val, test data
def data_preprocessing(filename = "liberals_vs_conservatives.csv"):
        raw_data = pd.read_csv(filename)
        raw_data.head()
        
        # We use the information of title and text to predict political lean
        data = raw_data[['Title', 'Text', 'Political Lean']]
        
        # preprocess the labels
        label_preprocessor = LabelEncoder()
        labels = torch.tensor(label_preprocessor.fit_transform(data['Political Lean'])).float()
        
        # preprocess the features
        features = data[['Title', 'Text']].fillna('')
        titles = features['Title']
        texts = features['Text']
        texts[texts != ''] = '\n' + texts[texts != '']
        texts = titles + texts
        
        # train val test split
        X_train, X_test, Y_train, Y_test = train_test_split(texts.values, labels, test_size = 0.2, random_state = 50)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 50, shuffle = False)
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        


# customed dataset

# tokenized dataset for pretrained model
class TokenizedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        super().__init__()
        self.texts = texts
        if type(labels) == torch.Tensor:
            self.labels = labels.float()
        else:
            self.labels = torch.Tensor(labels).float()
        self.tokenizer = tokenizer
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        one_input = self.tokenizer(self.texts[index], padding = "max_length", truncation = True, max_length = 128,
                                return_token_type_ids = True, return_tensors = 'pt')
        for key in one_input:
            one_input[key] = one_input[key].squeeze(dim = 0)
        one_input['labels'] = self.labels[index]
        return one_input


# dataset used for DAN
class IndicesDataset(Dataset):
    # can be used directly into the embedding layer
    def __init__(self, texts, labels, extractor):
        super().__init__()
        self.texts = texts
        if type(labels) == torch.Tensor:
            self.labels = labels.float()
        else:
            self.labels = torch.Tensor(labels).float()
        self.extractor = extractor
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        one_input = self.extractor.transform(self.texts[index])
    #    for key in one_input:
    #        one_input[key] = one_input[key].squeeze(dim = 0)
        one_input['labels'] = self.labels[index]
        return one_input

# extractors
class FeatureExtractor(Dataset):
    def __init__(self, remove_stopwords = True, padding = True):
        self.vocab = {}
        # the number of words in the vocabulary
        self.vocab_size = 0
        # maximum length of sentences
        self.max_length = 0
        self.remove_stopwords = remove_stopwords
        self.padding = padding
        
    # lower and lemmatize the words, remove stop words
    def clean(self, text):      
        # tokenize and lower
        words = word_tokenize(text.lower())
        
        word_list = []
        
        lemmatizer = WordNetLemmatizer()
        if self.remove_stopwords:
            for word in words:
                # remove all punctuations
                word = ''.join((char for char in word if char not in string.punctuation))
                # lemmatize
                word = lemmatizer.lemmatize(word)
                # remove stopwords
                if (word not in stopwords.words("english")) and word != '':
                    word_list.append(word)

        else:
            for word in words:
                # remove all punctuations
                word = ''.join((char for char in word if char not in string.punctuation))
                # lemmatize
                word = lemmatizer.lemmatize(word)
                if word != '':
                    word_list.append(word)
        return word_list
    
    # fit the model with training data
    def fit(self, texts):
        
        distinct_words = set()
        for text in texts:
            word_list = self.clean(text)
            distinct_words.update(set(word_list))
            self.max_length = max(self.max_length, len(word_list))
        
        for idx, word in enumerate(list(distinct_words)):
            self.vocab[word] = idx
        self.vocab_size = len(self.vocab)
        
        # add pad token and unk token
        self.vocab["<unk>"] = self.vocab_size
        self.vocab["<emp>"] = self.vocab_size + 1
        if self.padding:
            self.vocab["<pad>"] = self.vocab_size + 2
        # update the vocabulary size
        self.vocab_size = len(self.vocab)
        
    # transform text to features
    def transform(self, text):
        assert type(text) == str
        word_list = self.clean(text)
        features = []
        for word in word_list:
            index = self.vocab.get(word, self.vocab["<unk>"])
            features.append(index)
        
        # decide whether to pad (force the same length)
        if self.padding: # for DAN case
            # when no words left, add empty token to avoid error
            if len(features) == 0:
                features.append(self.vocab["<emp>"])
            # padding
            if len(features) < self.max_length:
                features += [self.vocab["<pad>"]] * (self.max_length - len(features))
            
            features = torch.Tensor(features).long()
            attention_mask = (features != self.vocab["<pad>"]).long()
            
            inputs = {"input_ids": features , "attention_mask": attention_mask}
        else: # for RNN case
            features = torch.Tensor(features).long()
            
            inputs = {"input_ids": features}
        
        
        return inputs
            
        
        
        
        
        