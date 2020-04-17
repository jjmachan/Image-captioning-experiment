import random
import time
import math

import torch
from torchvision import transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import rich

import aeye
from aeye.preprocessing import SOS_token, EOS_token, get_preprocessed_data, tensorForImageCaption
from aeye.trainUtils import asMinutes, timeSince

def train(
        img_tensor,
        sent_tensor,
        decoder,
        criterion,
        decoder_optimizer,
        max_length = 30,
        teacher_forcing_ratio = 1,
        ):

#     encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    length = sent_tensor.size(0)
    img_tensor = img_tensor.unsqueeze(0)
    sent_tensor = sent_tensor.unsqueeze(1)

    loss = 0

    decoder_input = SOS_tensor
    decoder_hidden = torch.cat([img_tensor,img_tensor], 0)
    decoder_c, _ = decoder.initHidden(1)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(length):
            decoder_output, decoder_hidden , decoder_c= decoder(
                                        decoder_input,
                                        decoder_hidden,
                                        decoder_c)

            #print('\n\n', decoder_output)
            #print('\n\n', decoder_hidden)

            loss += criterion(decoder_output[0], sent_tensor[di].squeeze(0))
            decoder_input = sent_tensor[di]

        #return decoder_output[0], sent_tensor

    else:
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_c = decoder(
                                        decoder_input,
                                        decoder_hidden,
                                        decoder_c)


            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze()
            loss += criterion(decoder_output[0], sent_tensor[0].squeeze(0))


    loss.backward()
    decoder_optimizer.step()

    return loss.item()/length

def trainIters(
        decoder,
        print_every=1000,
        plot_every=100,
        learning_rate=0.001
        ):

    plot_loss_total = 0
    print_loss_total = 0
    plot_losses = list()
    print_losses = list()
    start = time.time()

    # Get data
    feature_dict, sentence_list, lang = get_preprocessed_data('val')
    sentence_list = random.sample(sentence_list, len(sentence_list))
    n_iters = len(sentence_list)

    criterion = nn.NLLLoss().to(device)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    for iter, sentence in enumerate(sentence_list):
        img, sent = tensorForImageCaption(feature_dict, sentence, lang)
        img = img.to(device)
        sent = sent.to(device)
        loss = train(img, sent, decoder, criterion, decoder_optimizer)
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
    SOS_tensor = torch.tensor([[SOS_token]], device=device)
    EOS_tensor = torch.tensor([[EOS_token]], device=device)
    teacher_forcing_ratio = 1
    MAX_LENGTH = 30
    shuffle = True
    batch_size = 5
    num_workers = 1


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
                            transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = batch_size,
                                       shuffle = shuffle,
                                       num_workers = num_workers,
                                       collate_fn = aeye.collate_fn)

