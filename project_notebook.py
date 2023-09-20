#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader
#from tqdm import tqdm
from tqdm import tqdm

import transformers
from transformers import BertTokenizer, BertModel, AdamW

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(50)


# # New Section

# In[27]:


# from google.colab import drive
# drive.mount('/content/gdrive')
# raw_data = pd.read_csv("/content/gdrive/My Drive/data/liberals_vs_conservatives.csv")
raw_data = pd.read_csv("liberals_vs_conservatives.csv")
raw_data.head()


# In[28]:


# We use the information of title and text to predict political lean
data = raw_data[['Title', 'Text', 'Political Lean']]
labels, counts = np.unique(data['Political Lean'], return_counts = True)
print(labels, counts)


# In[29]:


# preprocess the labels
label_preprocessor = LabelEncoder()
labels = torch.tensor(label_preprocessor.fit_transform(data['Political Lean'])).float()
labels


# In[30]:


# preprocess the features by concatenate the two pieces of text with '\n' as the separater for those having text
features = data[['Title', 'Text']].fillna('')
titles = features['Title']
texts = features['Text']
texts[texts != ''] = '\n' + texts[texts != '']
texts = titles + texts


# In[31]:


lengths = texts.apply(lambda x: len(x))
plt.hist(lengths, bins = 50)
np.sum(lengths > 1024)/ len(lengths)
# cut about 5% of the instances


# In[32]:


# train val test split
X_train, X_test, Y_train, Y_test = train_test_split(texts.values, labels, test_size = 0.2, random_state = 50)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 50, shuffle = False)


# In[ ]:


# define the dataset for dataloader
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
    




# In[34]:


# Define the model
class BertPretrainedModel(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size = 10):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.batchNorm = nn.BatchNorm1d(hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        assert 'attention_mask' in inputs
        assert 'input_ids' in inputs
        # pooler output is the output for each entire sequence
        x = self.bert(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).pooler_output
        x = self.linear1(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.squeeze(dim = 1)
        prob = self.sigmoid(x)
        
        return prob


# In[40]:


# utility function
def average(L, weights):
    assert len(L) == len(weights) and len(L) == 2
    return (L[0] * weights[0] + L[1] * weights[1]) /(weights[0] + weights[1])

# training and evaluation for each epoch
def train_epoch(dataloader, model, criterion, optimizer, epoch):
    model.train()

    tqdmBar = tqdm(enumerate(dataloader), total = len(dataloader), 
                   desc = f"Training epoch {epoch:02d}", leave = True)
    # initialize metrics
    metrics_book = {'n': 0, 'loss': 0, 'accuracy': 0.0}
    for _, inputs in tqdmBar:
        labels = inputs['labels']
        N = len(labels)
        # zero gradient
        
        optimizer.zero_grad()

        # feedforward
        probs = model(inputs)
        preds = (probs > 0.5).float()

        # loss calculation and accuracy calculation
        loss = criterion(probs, labels)
        accuracy = torch.sum(preds == labels) / N

        # backpropagation
        loss.backward()
        
        # update
        optimizer.step()
        
        # book-keeping
        metrics_book['loss'] = average([loss, metrics_book['loss']], weights = [N, metrics_book['n']])
        metrics_book['accuracy'] = average([accuracy, metrics_book['accuracy']], weights = [N, metrics_book['n']])
        metrics_book['n'] += N
        
        # add information on the bar
        tqdmBar.set_postfix(loss = metrics_book['loss'], accuracy = metrics_book['accuracy'])

    return metrics_book
        
def val_epoch(dataloader, model, criterion, epoch):
    model.eval()
    with torch.no_grad():
        
        tqdmBar = tqdm(enumerate(dataloader), total = len(dataloader), 
                       desc = f" Evaluation epoch {epoch:02d}", leave = True)
        metrics_book = {'n': 0, 'loss': 0, 'accuracy': 0.0}
        
        for _, inputs in tqdmBar:
            labels = inputs['labels']
            N = len(labels)
            
            # feedforward
            probs = model(inputs)
            preds = (probs > 0.5).int()
            
            # loss and accuracy calculation
            loss = criterion(probs, labels)
            accuracy = torch.sum(preds == labels) / N
            
            # book-keeping
            metrics_book['loss'] = average([loss, metrics_book['loss']], weights = [N, metrics_book['n']])
            metrics_book['accuracy'] = average([accuracy, metrics_book['accuracy']], weights = [N, metrics_book['n']])
            metrics_book['n'] += N
            
            # add information on the bar
            tqdmBar.set_postfix(loss = metrics_book['loss'], accuracy = metrics_book['accuracy'])
        
    return metrics_book

def test(test_dataset, model):
    model.eval()
    with torch.no_grad():
        tqdmBar = tqdm(enumerate(test_dataset), total = len(test_dataset))
        preds = []
        labels = []
        for _, input in tqdmBar:
            label = input['labels']
            
            # feedforward
            prob = model(input)
            pred = int((prob > 0.5))
            
            # record
            preds.append(pred)
            labels.append(label)
    return preds, labels


# In[41]:


# Define the backbone function with the loss being recorded
def train_val(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs = 10):
    
    # initialize the tracker
    metrics_tracker = {'train_loss':[], 'train_accuracy':[],'val_loss':[], 'val_accuracy':[]}
    
    # main loop
    for epoch in range(1, num_epochs+1):

        train_metrics = train_epoch(train_dataloader, model, criterion, optimizer, epoch)
        val_metrics = val_epoch(val_dataloader, model, criterion, epoch)
        
        # update the tracker
        metrics_tracker['train_loss'].append(train_metrics['loss'])
        metrics_tracker['train_accuracy'].append(train_metrics['accuracy'])
        metrics_tracker['val_loss'].append(val_metrics['loss'])
        metrics_tracker['val_accuracy'].append(val_metrics['accuracy'])
    
    # visualize the performance curve
    
    # loss curve
    fig1, ax1 = plt.subplots(figsize = (12, 6))
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('number of epochs')
    ax1.set_ylabel('loss')
    ax1.plot(metrics_tracker['train_loss'], label = 'train', color = 'red')
    ax1.plot(metrics_tracker['val_loss'], label = 'val', color = 'magenta')
    ax1.grid()
    ax1.legend()
    
    # accuracy curve
    fig2, ax2 = plt.subplots(figsize = (12, 6))
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('number of epochs')
    ax2.set_ylabel('accuracy')
    ax2.plot(metrics_tracker['train_accuracy'], label = 'train', color = 'red')
    ax2.plot(metrics_tracker['val_accuracy'], label = 'val', color = 'magenta')
    ax2.grid()
    ax2.legend()


# In[42]:


# main script
model_name = 'bert-base-uncased'
batch_size = 16
bert_tokenizer  = BertTokenizer.from_pretrained(model_name)
train_dataset = TokenizedDataset(X_train[0:256], Y_train[0:256], bert_tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
val_dataset = TokenizedDataset(X_val[0:128], Y_val[0:128], bert_tokenizer)
val_dataloader = DataLoader(val_dataset, shuffle = False, batch_size = batch_size)
test_dataset = TokenizedDataset(X_test, Y_test, bert_tokenizer)
model = BertPretrainedModel(model_name)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr = 0.01)


# In[ ]:


train_val(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs = 10)


# In[ ]:


test(test_dataset, model)


# In[ ]:




