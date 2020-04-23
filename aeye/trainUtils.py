import time
import random
import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch import optim
from tqdm import tqdm

from .utils import convert_to_text
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

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


def sample(dataset, encoder, decoder, device, vocab):
    idx = random.randint(0, len(dataset))
    img_id, img, captions = dataset[idx]
    imgs  = img.unsqueeze(0).to(device)
    img_vec = encoder.to(device)(imgs)
    output = decoder.to(device).sample(img_vec)
    output_sent = convert_to_text(output, vocab)

    target_sents = list()
    for target in captions:
        target_sents.append(convert_to_text(target.tolist(), vocab))
    return (output_sent, target_sents)

