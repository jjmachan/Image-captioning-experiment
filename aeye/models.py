import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers = 2, device = None):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if device == None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.embeddings = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, c):
        output = self.embeddings(input)
        output = output.squeeze(2)
        output = F.relu(output)
        output, (hidden, c)= self.lstm(output, (hidden, c))
        output = self.out(output)
        #print(output.shape)
        output = self.softmax(output)
        #print(output.shape)
        return output, hidden, c

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

if __name__ == "__main__":
    hidden_size = 256
    output_size = 512
    batch_size = 20
    input = torch.zeros((1, batch_size, 1), dtype = torch.int64)
    decoder = DecoderLSTM(hidden_size, output_size)
    hidden, c = decoder.initHidden(batch_size)

    output, hidden, c = decoder(input, hidden, c)
    print(output.shape, hidden.shape, c.shape)

