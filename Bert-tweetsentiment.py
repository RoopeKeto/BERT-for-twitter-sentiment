#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import pandas as pd 
import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import logging
import re

# # # BUILDING THE BERT MODEL # # #

#first Activating logging
logging.basicConfig(level=logging.INFO)

# Loading pre-trained model tokenizer (vocabulary) using uncased version
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# testing out that bert tokenizer is working
text ='How are you today?'
text_tokenized = tokenizer.tokenize(text)
print(text_tokenized)

# normalizing the layer to reduce training time
class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        
# creating class for Bert sentiment classification task
class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


# configuration for the model
from pytorch_transformers import BertConfig

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

num_labels = 2

# creating the model
model = BertForSequenceClassification(num_labels)

# # # Testing model # # #
# Converting inputs to PyTorch tensors
tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(text_tokenized)])

logits = model(tokens_tensor)
# viewing the logits
print(logits)

# using softmax for classification
F.softmax(logits,dim=1)


# # # GET AND CLEAN THE DATA # # #

# the data has Sentiment column containing labels 1 = positive sentiment, 0 = negative
data = pd.read_csv('/home/roope/projects/sort-by-sentiment/training-data/Sentiment Analysis Dataset.csv',
                   error_bad_lines=False)

# taking only the first 400 000 examples to reduce training time
data = data[0:400000]

# removing whitespace 
data['SentimentText'] = data['SentimentText'].apply(str.strip)

# defining additional preprocessing function
def preprocessor(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
    
# applying preprocessor
data['SentimentText'] = data['SentimentText'].apply(preprocessor)

# shuffling the data
data = data.reindex(np.random.permutation(data.index))

# X and y variables from the data
X = data['SentimentText']  # = tekstinä
y = data['Sentiment'] # = 0 / 1 arvot

# splitting data into train and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

y_train = pd.get_dummies(y_train).values.tolist()
y_test = pd.get_dummies(y_test).values.tolist()

# creating the text_dataset class
max_seq_length = 128
class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):
        
        self.x_y_list = x_y_list
        self.transform = transform
        
    def __getitem__(self,index):
        #tokenization
        tokenized_tweet = tokenizer.tokenize(self.x_y_list[0][index])
        # to fit into memory and speed, applying max_seq_length
        if len(tokenized_tweet) > max_seq_length:
            tokenized_tweet = tokenized_tweet[:max_seq_length]
            
        ids_tweet  = tokenizer.convert_tokens_to_ids(tokenized_tweet)

        # padding with zeros if necessary
        padding = [0] * (max_seq_length - len(ids_tweet))
        
        ids_tweet += padding
        
        assert len(ids_tweet) == max_seq_length
        
        # making into torch tensor
        ids_tweet = torch.tensor(ids_tweet)
        
        sentiment = self.x_y_list[1][index]         
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_tweet, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])

# defining the options in training
batch_size = 16

train_lists = [X_train, y_train]
test_lists = [X_test, y_test]

training_dataset = text_dataset(x_y_list = train_lists )

test_dataset = text_dataset(x_y_list = test_lists )

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

# enabling gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# defining the training model. 
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            sentiment_corrects = 0
            
            
            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
                #inputs = inputs
                #print(len(inputs),type(inputs),inputs)
                #inputs = torch.from_numpy(np.array(inputs)).to(device) 
                inputs = inputs.to(device) 

                sentiment = sentiment.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs)

                    outputs = F.softmax(outputs,dim=1)
                    
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(phase, sentiment_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test.pth')


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# using gpu
model.to(device)

# defining hyperparameters
# Using different learning rate for the Bert part / main model (smaller) and for the added model parts / weights
lrlast = .001
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},
       
   ])

# Observe that all parameters are being optimized
optimizer_ft = optim1
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)


# # # TRAINING THE MODEL # # #
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)


# # # RESULTS # # #
# After 4 epochs on data of 400 000 (training size 360 000)
# train accuracy 83,99 % and test set accuracy of 81,4 %
# this training took on rtx2070 gpu about 8,5 hours
# saved parameters in bert_model_test.pth



