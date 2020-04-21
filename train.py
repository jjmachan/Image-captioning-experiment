import random
import time
import math

import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import rich
from tqdm import tqdm

import aeye
from aeye import utils
from aeye.trainUtils import trainIters, eval_loss
from aeye.preprocessing import collate_fn
from aeye.trainUtils import asMinutes, timeSince
from aeye.models import Encoder, DecoderLSTM


def main():
    # Configs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shuffle = True
    batch_size = 10
    num_workers = 0
    epochs = 300
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
                                       batch_size = batch_size,
                                       shuffle = shuffle,
                                       num_workers = num_workers,
                                       collate_fn = collate_fn)

    testDataloader = torch.utils.data.DataLoader(dataset = dataset_test,
                                       batch_size = batch_size,
                                       shuffle = shuffle,
                                       num_workers = num_workers,
                                       collate_fn = collate_fn)

    # valDataloader will not work because the sentences are not padded
    # and are of different lengths.


    # INITIATE MODELS
    # each sample in the dataset has 5 sentences and each will be used for
    # training the model
    decoder = DecoderLSTM(hidden_size, vocab.n_words, batch_size*5, device)
    encoder = Encoder(hidden_size)
    criterion = nn.CrossEntropyLoss()

    losses_train = list()
    losses_val = list()

    try:
        for epoch in range(epochs):
            # train 1 epoch
            print('[Epoch: %d / %d]'%(epoch+1, epochs))
            print_losses, plot_losses = trainIters(trainDataloader,
                                                   encoder.to(device),
                                                   decoder.to(device),
                                                   device,
                                                   criterion,
                                                   print_every=100)

            print('[Epoch] Training Loss: %.4f'%(sum(print_losses)/len(print_losses)))
            losses_train += print_losses

            print('[Epoch] Running Eval')
            eval_losss = eval_loss(testDataloader,
                                    encoder.to(device),
                                    decoder.to(device),
                                    criterion,
                                    device)
            print('[Epoch] Eval loss: %.4f'%(eval_losss))
            losses_val.append(eval_losss)

    except KeyboardInterrupt:
        pass
    finally:
        utils.save_model(encoder, decoder, epoch, losses_train, losses_val, \
                'saved_models')
if __name__ == '__main__':
    main()
