# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:04:19 2022

@author: Xxp
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader


import transformers
from transformers import BertTokenizer, BertModel, AdamW

# import other files
from dataset import *
from models import *
from train_val_test import *

torch.manual_seed(50)

# main script

# main script
def run(return_model_for_test = False, model_type = 'dan', learning_rate = 0.005, 
        num_epochs = 20, embedding_dim = 200, hidden_dim = 20, dropout_rate = 0.2, 
        num_layers = 2, first_hidden_dim = 100, second_hidden_dim = 50, num_heads = 4):
    # device detect
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # hyperparameters irrelavent to the class
    # model_type = 'dan'
    # learning_rate = 0.005
    # num_epochs = 20

    if model_type == 'bert':
        # hidden_dim = 50
        # dropout_rate = 0.2
        # create the model
        model_name = 'bert-base-uncased'
        model = BertPretrainedModel(model_name, hidden_dim = hidden_dim, dropout_rate = dropout_rate)
        # create the tokenizer
        bert_tokenizer  = BertTokenizer.from_pretrained(model_name)
        
        
        # create the dataset
        train_dataset = TokenizedDataset(X_train, Y_train, bert_tokenizer)
        val_dataset = TokenizedDataset(X_val, Y_val, bert_tokenizer)
        test_dataset = TokenizedDataset(X_test, Y_test, bert_tokenizer)
        
        # hidden_dim = 50
        # dropout_rate = 0.2
        # create the model
        model_name = 'bert-base-uncased'
        model = BertPretrainedModel(model_name, hidden_dim = hidden_dim, dropout_rate = dropout_rate)
        # create the tokenizer
        bert_tokenizer  = BertTokenizer.from_pretrained(model_name)
        
        
        # create the dataset
        train_dataset = TokenizedDataset(X_train, Y_train, bert_tokenizer)
        val_dataset = TokenizedDataset(X_val, Y_val, bert_tokenizer)
        test_dataset = TokenizedDataset(X_test, Y_test, bert_tokenizer)

    elif model_type == 'dan':
        # create the feature extractor
        feature_extractor = FeatureExtractor()
        feature_extractor.fit(X_train)
        
        # create the model
        # embedding_dim = 200
        # hidden_dim = 50
        # dropout_rate = 0.2
        model = DANModel(feature_extractor.vocab_size, embedding_dim, hidden_dim, dropout_rate = dropout_rate)
        # create the dataset
        train_dataset = IndicesDataset(X_train, Y_train, feature_extractor)
        val_dataset = IndicesDataset(X_val, Y_val, feature_extractor)
        test_dataset = IndicesDataset(X_test, Y_test, feature_extractor)

    elif model_type == 'lstm':
            
            # create the feature extractor
            feature_extractor = FeatureExtractor(padding = False, remove_stopwords = False)
            feature_extractor.fit(X_train)
            
            # create the model
            
            # embedding_dim = 100
            # hidden_dim = 20
            # num_layers = 2
            # dropout_rate = 0.2
            model = BiLSTMModel(feature_extractor.vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate = dropout_rate)
            # create the dataset
            train_dataset = IndicesDataset(X_train, Y_train, feature_extractor)
            val_dataset = IndicesDataset(X_val, Y_val, feature_extractor)
            test_dataset = IndicesDataset(X_test, Y_test, feature_extractor)
    
    elif model_type == 'gru':
            
        # create the feature extractor
        feature_extractor = FeatureExtractor(padding = False, remove_stopwords = False)
        feature_extractor.fit(X_train)
        
        # create the model
        
        # embedding_dim = 100
        # hidden_dim = 20
        # num_layers = 2
        # dropout_rate = 0.2
        model = BiGRUModel(feature_extractor.vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate = dropout_rate)
        # create the dataset
        train_dataset = IndicesDataset(X_train, Y_train, feature_extractor)
        val_dataset = IndicesDataset(X_val, Y_val, feature_extractor)
        test_dataset = IndicesDataset(X_test, Y_test, feature_extractor)
    
    elif model_type == 'bert_attention_gru': 
        # first_hidden_dim = 100
        # second_hidden_dim = 50
        # num_heads = 4
        # dropout_rate = 0.2
        # create the model
        model_name = 'bert-base-uncased'
        model = BertAttentionGRUModel(first_hidden_dim, second_hidden_dim, num_heads = 4, 
                     dropout_rate = 0.2, pretrained_model_name = 'bert_base_uncased')
        # create the tokenizer
        bert_tokenizer  = BertTokenizer.from_pretrained(model_name)
        
        
        # create the dataset
        train_dataset = TokenizedDataset(X_train, Y_train, bert_tokenizer)
        val_dataset = TokenizedDataset(X_val, Y_val, bert_tokenizer)
        test_dataset = TokenizedDataset(X_test, Y_test, bert_tokenizer)
    
    
    # from dataloader to model input    
    if model_type in ['lstm', 'gru', 'bert_attention_gru']:
        batch_size = 1
    else:
        batch_size = 16
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle = False, batch_size = batch_size)

    # create criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)

    # training and evaluation

    val_scores = train_val(train_dataloader, val_dataloader, val_dataset, model, criterion, optimizer, device = device, num_epochs = num_epochs)

    if return_model_for_test == 'True':
        return val_scores, model, test_dataset
    else:
        return val_scores


    
if __name__ == "__main__":
    run(return_model_for_test = False, model_type = 'bert_attention_gru', learning_rate = 3e-5, 
        num_epochs = 5, dropout_rate = 0.2, first_hidden_dim = 100, 
        second_hidden_dim = 50, num_heads = 4)