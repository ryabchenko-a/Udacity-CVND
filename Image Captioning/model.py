import torch
import torch.nn as nn
import torchvision.models as models
from math import log
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # embed captions and concatenate with features along dim=1
        # last caption word (end) is not embedded, since we don't pass it to the RNN
        inputs = torch.cat([features.unsqueeze(1), self.embed(captions[:,:-1])], 1)
        out, hidden = self.lstm(inputs)
        outputs = self.linear(out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        # accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        was_training = self.training
        self.eval()
        with torch.no_grad():
        
            sentence = []
            last_word = 0
            i = 0
            hidden = None
                          
            while (i < max_len) & (last_word != 1):
                out, hidden = self.lstm(inputs, hidden)
                pred = self.linear(out).squeeze()
                last_word = torch.argmax(pred).item()
                sentence.append(last_word)
                inputs = self.embed(torch.argmax(pred)).view(1, 1, -1)
                
                i += 1
        
        if was_training:
            self.train()
            
        return sentence
    
    def beam_search(self, inputs, width=10, max_len=20):
        
        was_training = self.training
        self.eval()
        with torch.no_grad():
            
            sentences = []
            out, hidden = self.lstm(inputs)
            pred = self.linear(out).squeeze()
            probable_words = np.argsort(torch.squeeze(pred).cpu().numpy())[::-1][:width].tolist()
            
            # initialize the beams
            for i in range(width):
                sentences.append([[probable_words[i]], hidden, log(pred[probable_words[i]]), 1])
            last_words = probable_words
            
            # search
            options = []
            stop_criterion = [1]*width
            l = 1
            
            # implement beam search
            while (last_words != stop_criterion) & (l < max_len): # either all sentences are complete or max length is reached
                
                # loop through each sentence to construct options 
                # [width] options for each sentence - max [width]*[width] option
                for i in range(width):  
                    sentence = sentences[i][0]
                    last_word = torch.Tensor([sentence[-1]]).type(torch.cuda.LongTensor)
                    
                    if last_word == 1:
                        continue
                    
                    hidden = sentences[i][1]
                    prob_log = sentences[i][2]
                    num_words = sentences[i][3]
                    
                    ### embed last word in a sentence to pass as an input
                    inputs = self.embed(last_word).view(1, 1, -1)
                        
                    ### find the most probable next words
                    out, hidden = self.lstm(inputs, hidden)
                    pred = self.linear(out).squeeze()
                    probable_words = np.argsort(torch.squeeze(pred).cpu().numpy())[::-1][:width].tolist() # you may use torch.argsort() and do not convert tensor to numpy array. Unfortunately, Udacity's Pytorch is old version and does not have argsort.
                    ### add the found options
                    for j in range(width):
                        new_sentence = sentence + [probable_words[j]]
                        options.append([new_sentence, hidden, prob_log+log(pred[probable_words[j]]), num_words+1])

                ### sort by sum(log(p))/n and choose the [width] highest ranked options
                options_sorted = sorted(options, key=lambda option: -option[2]/option[3])
                sentences = options_sorted[:width]
                last_words = list(map(lambda x: x[0][-1], sentences))
                
                l += 1
                options = []
        
        if was_training:
            self.train()
            
        ### only the sentences are returned
        sentences = list(map(lambda x: x[0], sentences))
            
        return sentences