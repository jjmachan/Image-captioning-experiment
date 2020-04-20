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
from tqdm import tqdm

import aeye
from aeye.preprocessing import collate_fn
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
        criterion,
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

    criterion = criterion.to(device)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    for iter, (_, imgs, captions, caption_lengths) in enumerate(dataloader):
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

def eval_loss(dataloader,
         encoder,
         decoder,
         criterion,
         device):
    batch_losses = list()
    loss = 0

    for i, (img_ids, imgs, captions, caption_lengths) in tqdm(enumerate(dataloader), desc=loss, total=len(dataloader)):
        with torch.no_grad():
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = torch.tensor(caption_lengths).to(device)
            img_vecs = encoder(imgs)
            output, _ = decoder(img_vecs, captions.t(), caption_lengths)
            target = pack_padded_sequence(captions.t(), caption_lengths).data

            loss = criterion(output, target)
            batch_losses.append(loss.item())

    return sum(batch_losses)/len(batch_losses)

if __name__ == '__main__':
    # Configs
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    shuffle = True
    batch_size = 5
    num_workers = 1
    epochs = 1
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

    dataset_train = aeye.Flickr8k(img_dir=img_files,
                            ann_file=ann_file,
                            vocab=vocab,
                            split='train',
                            transform=transform)

    dataset_test = aeye.Flickr8k(img_dir=img_files,
                            ann_file=ann_file,
                            vocab=vocab,
                            split='test',
                            transform=transform)

    dataset_val = aeye.Flickr8k(img_dir=img_files,
                            ann_file=ann_file,
                            vocab=vocab,
                            split='val',
                            transform=transform)

    trainDataloader = torch.utils.data.DataLoader(dataset = dataset_train,
                                       batch_size = 10,
                                       shuffle = True,
                                       num_workers = 3,
                                       collate_fn = collate_fn)

    testDataloader = torch.utils.data.DataLoader(dataset = dataset_test,
                                       batch_size = 10,
                                       shuffle = True,
                                       num_workers = 1,
                                       collate_fn = collate_fn)

    # valDataloader will not work because the sentences are not padded
    # and are of different lengths.


    # INITIATE MODELS
    # each sample in the dataset has 5 sentences and each will be used for
    # training the model
    decoder = DecoderLSTM(hidden_size, vocab.n_words, batch_size*5, device)
    encoder = Encoder(hidden_size)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        # train 1 epoch
        print('[Epoch: %d / %d]'%(epoch+1, epochs))
        print_losses, plot_losses = trainIters(trainDataloader,
                                               encoder.to(device),
                                               decoder.to(device),
                                               device,
                                               criterion,
                                               print_every=10)

        print('[Epoch] Training Loss: %.4f'%(sum(print_losses)/len(print_losses)))

        print('[Epoch] Running Eval')
        eval_losss = eval_loss(testDataloader,
                                encoder.to(device),
                                decoder.to(device),
                                criterion,
                                device)
        print('[Epoch] Eval loss: %.4f'%(eval_losss))
