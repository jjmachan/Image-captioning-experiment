import os
from datetime import datetime

import torch

from .preprocessing import Lang, PAD_token, SOS_token, EOS_token

def convert_to_text(sample_ids: list, vocab: Lang):
    words = [vocab.index2word[word_id] for word_id in sample_ids]
    return ' '.join(words)

def save_model(encoder, decoder, epoch, losses_train, losses_val, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    print('Saving model to: ',checkpoint_path)

    checkpoint_file = os.path.join(checkpoint_path, 'model-%d-%s.ckpt'\
            %(epoch+1, datetime.now()))
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'epoch': epoch,
        'losses_train': losses_train,
        'losses_val': losses_val
        }, checkpoint_file)

def load_model(checkpoint_file, sample=False):
    checkpoint = torch.load(checkpoint_file)

    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    epoch = checkpoint['epoch']
    losses_train  = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']

    return encoder_state_dict, decoder_state_dict, epoch, losses_train, \
            losses_val
