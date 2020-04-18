import random
import time
import math

import torch
from torchvision import transforms
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import rich

import aeye
from aeye.trainUtils import asMinutes, timeSince
from aeye.models import Encoder, DecoderLSTM

def train(
        img_tensor,
        caption_tensor,
        caption_lengths,
        encoder,
        decoder,
        criterion,
        encoder_optimizer = None,
        decoder_optimizer = None,
        max_length = 30,
        ):

    # Encode image
    if encoder_optimizer is not None:
        encoder_optimizer.zero_grad()
        img_vec = encoder(img_tensor)
    else:
        with torch.no_grad():
            img_vec = encoder(img_tensor)

    decoder_optimizer.zero_grad()


    # Decoder
    output, _ = decoder(img_vec, caption_tensor.t(), caption_lengths)

    target = pack_padded_sequence(caption_tensor.t(), caption_lengths).data
    loss = criterion(output, target)

    loss.backward()
    if encoder_optimizer is not None:
        encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def trainIters(
        dataloader,
        encoder,
        decoder,
        device,
        print_every=1000,
        plot_every=100,
        learning_rate=0.001
        ):

    plot_loss_total = 0
    print_loss_total = 0
    plot_losses = list()
    print_losses = list()
    start = time.time()
    # the number of batches
    n_iters = len(dataloader)

    criterion = nn.CrossEntropyLoss().to(device)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    for iter, (_, imgs, captions, caption_lengths) in enumerate(dataloader, start=0):
        imgs = imgs.to(device)
        captions = captions.to(device)
        caption_lengths = torch.tensor(caption_lengths).to(device)
        loss = train(imgs, captions, caption_lengths, encoder, decoder, criterion, decoder_optimizer=decoder_optimizer)
        print_loss_total += loss

        if (iter+1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_losses.append(print_loss_avg)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                        iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return print_losses, plot_losses



if __name__ == '__main__':
    # Configs
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    shuffle = True
    batch_size = 5
    num_workers = 1
    epochs = 10
    hidden_size = 512

    # LOAD DATA
    vocab = aeye.load_vocab('train_vocab.pkl')

    ann_file = '/home/jithin/datasets/imageCaptioning/captions/dataset_flickr8k.json'
    img_files = '/home/jithin/datasets/imageCaptioning/flicker8k/Flicker8k_Dataset/'

    transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
     )])

    dataset = aeye.Flickr8k(img_dir=img_files,
                            ann_file=ann_file,
                            vocab=vocab,
                            split='val',
                            transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = batch_size,
                                       shuffle = shuffle,
                                       num_workers = num_workers,
                                       collate_fn = aeye.collate_fn)


    # INITIATE MODELS
    # each sample in the dataset has 5 sentences and each will be used for
    # training the model
    decoder = DecoderLSTM(hidden_size, vocab.n_words, batch_size*5, device)
    encoder = Encoder(hidden_size)



    for epoch in range(epochs):
        # train 1 epoch
        print('[Epoch: %d / %d]'%(epoch+1, epochs))
        print_losses, plot_losses = trainIters(dataloader,
                                               encoder.to(device),
                                               decoder.to(device),
                                               device,
                                               print_every=100)

        print('[Epoch] Training Loss: %.4f'%(sum(print_losses)/len(print_losses)))
