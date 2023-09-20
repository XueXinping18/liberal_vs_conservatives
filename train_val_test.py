# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:58:45 2022

@author: Xxp
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


torch.manual_seed(50)

# utility function used
def average(L, weights):
    assert len(L) == len(weights) and len(L) == 2
    return (L[0] * weights[0] + L[1] * weights[1]) /(weights[0] + weights[1])


# training and evaluation for each epoch
def train_epoch(dataloader, model, criterion, optimizer, epoch, device):
    model.train()

    tqdmBar = tqdm(enumerate(dataloader), total = len(dataloader), 
                   desc = f"Training epoch {epoch:02d}", leave = True)
    # initialize metrics
    metrics_book = {'n': 0, 'loss': 0, 'accuracy': 0.0}
    for _, inputs in tqdmBar:

        # fetch data to gpu
        for key in inputs:
            inputs[key] = inputs[key].to(device, non_blocking = True)

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
        metrics_book['loss'] = average([loss.item(), metrics_book['loss']], weights = [N, metrics_book['n']])
        metrics_book['accuracy'] = average([accuracy.item(), metrics_book['accuracy']], weights = [N, metrics_book['n']])
        metrics_book['n'] += N
        
        # add information on the bar
        tqdmBar.set_postfix(loss = metrics_book['loss'], accuracy = metrics_book['accuracy'])

    return metrics_book
        
def val_epoch(dataloader, model, criterion, epoch, device):
    model.eval()
    with torch.no_grad():
        
        tqdmBar = tqdm(enumerate(dataloader), total = len(dataloader), 
                       desc = f" Evaluation epoch {epoch:02d}", leave = True)
        metrics_book = {'n': 0, 'loss': 0, 'accuracy': 0.0}
        
        for _, inputs in tqdmBar:

            # fetch data to gpu
            for key in inputs:
                inputs[key] = inputs[key].to(device, non_blocking = True)

            # record information
            labels = inputs['labels']
            N = len(labels)
            
            # feedforward
            probs = model(inputs)
            preds = (probs > 0.5).int()
            
            # loss and accuracy calculation
            loss = criterion(probs, labels)
            accuracy = torch.sum(preds == labels) / N
            
            # book-keeping
            metrics_book['loss'] = average([loss.item(), metrics_book['loss']], weights = [N, metrics_book['n']])
            metrics_book['accuracy'] = average([accuracy.item(), metrics_book['accuracy']], weights = [N, metrics_book['n']])
            metrics_book['n'] += N
            
            # add information on the bar
            tqdmBar.set_postfix(loss = metrics_book['loss'], accuracy = metrics_book['accuracy'])

    return metrics_book

def scores(y_true, y_label):
    scores = {}
    scores['precision'] = precision_score(y_true, y_label)
    scores['recall'] = recall_score(y_true, y_label)
    scores['f1_score'] = f1_score(y_true, y_label)
    return scores

def test(test_dataset, model, device):
    model.eval()
    with torch.no_grad():
        tqdmBar = tqdm(enumerate(test_dataset), total = len(test_dataset))
        preds = []
        labels = []
        for _, one_input in tqdmBar:
            for key in one_input:
                one_input[key] = one_input[key].to(device, non_blocking = True).unsqueeze(dim = 0)
            label = one_input['labels'].item()
            
            # feedforward
            prob = model(one_input).item()
            pred = float(int((prob > 0.5)))
            
            # record
            preds.append(pred)
            labels.append(label)
            # calculate all the scores
            score_table = scores(labels, preds)

    return preds, labels, score_table

# Define the backbone function with the best loss being recorded
def train_val(train_dataloader, val_dataloader, val_dataset, model, criterion, optimizer, device = 'cpu', num_epochs = 10):
    
    # initialize the tracker
    metrics_tracker = {'train_loss':[], 'train_accuracy':[],'val_loss':[], 'val_accuracy':[]}
    # fetch model to gpu
    model = model.to(device)
    # main loop
    for epoch in range(1, num_epochs+1):

        train_metrics = train_epoch(train_dataloader, model, criterion, optimizer, epoch, device)
        val_metrics = val_epoch(val_dataloader, model, criterion, epoch, device)
        
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
    plt.savefig("loss_curve.png")
    
    # accuracy curve
    fig2, ax2 = plt.subplots(figsize = (12, 6))
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('number of epochs')
    ax2.set_ylabel('accuracy')
    ax2.plot(metrics_tracker['train_accuracy'], label = 'train', color = 'red')
    ax2.plot(metrics_tracker['val_accuracy'], label = 'val', color = 'magenta')
    ax2.grid()
    ax2.legend()
    plt.savefig("accuracy_curve.png")
    
    # obtain precision, recall, f1 scores for the final validation

    accuracy_final = metrics_tracker['val_accuracy'][-1]
    _,_,score_dict = test(val_dataset, model, device)
    score_dict['accuracy'] = accuracy_final
    return score_dict