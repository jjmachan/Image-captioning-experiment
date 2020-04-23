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
    def __init__(self, hidden_size, output_size, batch_size, device, num_layers = 1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.embeddings = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, img_vec, captions, lengths):
        hidden = self.initHidden(img_vec)
        output = self.embeddings(captions)
        output = F.relu(output)
        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output.data)

        return output, hidden

    def sample(self, img_vec, max_length=30):
        samples = list()

        # Check if image batch_size is 1
        assert img_vec.size(0) == 1
        hidden = self.initHidden(img_vec, batch_size=1)
        input_cap = torch.tensor([1], device=self.device).long().unsqueeze(0)
        for i in range(max_length):
            input = self.embeddings(input_cap)
            input = F.relu(input)
            output, hidden = self.lstm(input, hidden)
            output = self.out(output)

            input_cap = output.topk(1).indices
            input_cap = input_cap.squeeze(0)
            samples.append(input_cap.item())


        return samples

    def initHidden(self, img_vec, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        img_vec = img_vec.unsqueeze(0)
        assert img_vec.shape == (self.num_layers, batch_size, self.hidden_size)
        return (img_vec, torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

class DecoderLSTM_mod(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, device, num_layers = 1):
        super(DecoderLSTM_mod, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.embeddings = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, img_vec, captions, lengths):
        hidden = self.initHidden(img_vec)
        output = self.embeddings(captions)
        output = F.relu(output)

        lengths = lengths - 1
        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output.data)

        return output, hidden

    def sample(self, img_vec, max_length=30):
        samples = list()

        # Check if image batch_size is 1
        assert img_vec.size(0) == 1
        hidden = self.initHidden(img_vec, batch_size=1)
        input_cap = torch.tensor([1], device=self.device).long().unsqueeze(0)

        for i in range(max_length):
            input = self.embeddings(input_cap)
            input = F.relu(input)
            output, hidden = self.lstm(input, hidden)
            output = self.out(output)

            input_cap = output.topk(1).indices
            input_cap = input_cap.squeeze(0)
            samples.append(input_cap.item())


        return samples

    def initHidden(self, img_vec, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        img_vec = img_vec.unsqueeze(0)

        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        h = img_vec.expand_as(c).contiguous()
        return (h,c)

if __name__ == "__main__":
    hidden_size = 256
    output_size = 512
    batch_size = 20
    input = torch.zeros((1, batch_size, 1), dtype = torch.int64)
    decoder = DecoderLSTM(hidden_size, output_size)
    hidden, c = decoder.initHidden(batch_size)

    output, hidden, c = decoder(input, hidden, c)
    print(output.shape, hidden.shape, c.shape)

