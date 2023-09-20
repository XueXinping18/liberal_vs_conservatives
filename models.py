# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:35:54 2022

@author: Xxp
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertTokenizer, BertModel, AdamW



torch.manual_seed(50)

# bert pretrained model
class BertPretrainedModel(nn.Module):
    def __init__(self, pretrained_model_name = 'bert_base_uncased'):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name, use_auth_token=True)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        assert 'attention_mask' in inputs
        assert 'input_ids' in inputs
        # pooler output is the output for each entire sequence
        x = self.bert(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).pooler_output
        x = self.linear1(x)
        x = x.squeeze(dim = 1)
        prob = self.sigmoid(x)
        
        return prob

# Deep average network
class DANModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.batchNorm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)

        
    def forward(self, inputs):
        
        x = self.embedding(inputs["input_ids"])
        x[inputs["attention_mask"]] = float('nan')
        x = torch.nanmean(x, dim = -2)
        x = self.linear1(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.squeeze(dim = 1)
        probs = torch.sigmoid(x)

        return probs

# BiLSTM network
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 20, num_layers = 1, dropout_rate = 0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers, 
                            batch_first = True, bidirectional = True)
        self.feedforward = nn.Sequential(nn.Dropout(p = dropout_rate),
                                     nn.Linear(hidden_dim*num_layers*2, hidden_dim//2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim//2, 1),
                                     nn.Sigmoid())
        

    def forward(self, inputs):
        x = self.embedding(inputs["input_ids"])
        # only final hidden states used as predictor
        _, (final_hidden, final_cell) = self.lstm(x)
        x = torch.transpose(final_hidden, 0, 1)
        x = x.view(x.shape[0], -1)
        probs = self.feedforward(x).squeeze(dim = 1)

        return probs

# biGRU network
class BiGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 20, num_layers = 1, dropout_rate = 0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers = num_layers, 
                            batch_first = True, bidirectional = True)
        self.feedforward = nn.Sequential(nn.Dropout(p = dropout_rate),
                                     nn.Linear(hidden_dim*num_layers*2, hidden_dim//2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim//2, 1),
                                     nn.Sigmoid())
        

    def forward(self, inputs):
        x = self.embedding(inputs["input_ids"])
        # only final hidden states used as predictor
        _, (final_hidden, final_cell) = self.gru(x)
        x = torch.transpose(final_hidden, 0, 1)
        x = x.view(x.shape[0], -1)
        probs = self.feedforward(x).squeeze(dim = 1)

        return probs

# multihead self attention model 
class BertAttentionGRUModel(nn.Module):
    def __init__(self,first_hidden_dim , second_hidden_dim, num_heads = 4, 
                 dropout_rate = 0.2, pretrained_model_name = 'bert_base_uncased'):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name, use_auth_token=True)
        # build a multihead attention
        self.keyLinear = nn.Linear(self.bert.config.hidden_size, first_hidden_dim, bias = False)
        self.queryLinear = nn.Linear(self.bert.config.hidden_size, first_hidden_dim, bias = False)
        self.valueLinear = nn.Linear(self.bert.config.hidden_size, first_hidden_dim, bias = False)
        self.multiheadAttention = nn.MultiheadAttention(first_hidden_dim, num_heads, dropout = dropout_rate, batch_first = True)
        self.gru = nn.GRU(first_hidden_dim, second_hidden_dim, batch_first = True, bidirectional = True)
        self.linear = nn.Linear(second_hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        assert 'attention_mask' in inputs
        assert 'input_ids' in inputs
        # we need final hidden layers of bert output at each states
        x = self.bert(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state
        keys = self.keyLinear(x)
        queries = self.queryLinear(x)
        values = self.valueLinear(x)
        x = self.multiheadAttention(queries, keys, values)
        _, (final_hidden, final_cell) = self.gru(x)
        x = torch.transpose(final_hidden, 0, 1)
        x = x.view(x.shape[0], -1) 
        probs = self.sigmoid(self.linear(x)).squeeze(dim = 1)
        
        return probs
# Idea: Text Classification Model Based on Multi-head self-attention mechanism and BiGRU