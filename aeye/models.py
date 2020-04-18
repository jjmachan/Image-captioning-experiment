import random

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """Class to build new model including all but last layers"""
    def __init__(self, output_dim=1000):
        super(Encoder, self).__init__()
        # TODO: change with resnet152?
        pretrained_model = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(pretrained_model.fc.in_features, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        # weight init, inspired by tutorial
        self.linear.weight.data.normal_(0,0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.linear(x)

        return x

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, num_layers = 1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embeddings = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, img_vec, captions, lengths):
        hidden = self.initHidden(img_vec)
        output = self.embeddings(captions)
        output = F.relu(output)
        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths)
        output, hidden= self.lstm(output, hidden)
        output = self.out(output.data)

        return output, hidden

    def initHidden(self, img_vec):
        img_vec = img_vec.unsqueeze(0)
        assert img_vec.shape == (self.num_layers, self.batch_size, self.hidden_size)
        return (img_vec,
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


if __name__ == "__main__":
    hidden_size = 256
    output_size = 512
    batch_size = 20
    input = torch.zeros((1, batch_size, 1), dtype = torch.int64)
    decoder = DecoderLSTM(hidden_size, output_size)
    hidden, c = decoder.initHidden(batch_size)

    output, hidden, c = decoder(input, hidden, c)
    print(output.shape, hidden.shape, c.shape)

