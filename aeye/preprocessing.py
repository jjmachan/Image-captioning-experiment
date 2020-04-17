from pickle import dump
from pickle import load
from pathlib import Path
import json
import pickle

from torchvision import models
from torchvision import transforms
import torch
from torch import nn
from tqdm import tqdm

from .datasets import Flickr8k

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK = 3
        self.word2index = {"UNK": 3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS


#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __call__(self, word):
        if not word in self.word2index:
            return self.word2index['UNK']
        return self.word2index[word]

    def __len__(self):
        return self.n_words

def make_vocab(vocab_name, ann_file = None):
    if not ann_file == None:
        eng = Lang(vocab_name)
        with open(ann_file) as f:
            for image in tqdm(json.load(f)['images']):
                for sentence in image['sentences']:
                    for word in sentence['tokens']:
                        eng.addWord(word)
        print('Vocab is generated!')

        with open('%s.pkl'%(vocab_name), 'wb') as f:
            pickle.dump(eng, f)
    else:
        print('Please provide a valid annotation file')

def load_vocab(path):
    with open(path, 'rb') as f:
        lang = pickle.load(f)
    return lang

def gen_img_encodings(model, dataset):
    """
    Generate the encodding for the images.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Force to cpu due to memory issue in gpu
    # TODO: fix memory issue in gpu
    #device = torch.device('cpu')
    features = dict()
    model.eval()
    model.to(device)
    with torch.no_grad():
        for img_id, img, sentences in tqdm(dataset):
            img = torch.unsqueeze(img, 0).to(device)
            feature = model(img)
            features[img_id] = feature.cpu().numpy()
            del img,feature
    return features

def gen_sentence_lists(dataset, language_name):
    """
    Generate language and returns the sentence list with the corresponding
    image ids.
    """
    lang = Lang(language_name)
    sentences_list = list()

    for idx, _, sentences in tqdm(dataset):
        for sentence in sentences:
            for word in sentence:
                lang.addWord(word)

            sentences_list.append((idx, sentence))
    return sentences_list,lang

def indexesFromSentence(lang: Lang, sentence: list):
    return [lang.word2index[word] for word in sentence]

def tensorFromSentence(lang, sentence, device):
    if device == None:
        device = torch.device('cpu')
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorForImageCaption(features_dict, sentence_tuple, lang, batch_size, device = None):
    """
    This created the input_img, imput_vec pair that is used for training
    the model.

    Args:
        sentence_tuple: tuple with the img idx and tokenized sentence
        feature_dict: dict that maps the img idx and the feature tensor
        lang: The language model used.
    """
    idx, sentence = sentence_tuple
    target_tensor = tensorFromSentence(lang, sentence, device)
    features  = features_dict[idx]
    features = torch.from_numpy(features)
    return features, target_tensor

def get_preprocessed_data(split: str, data_path = None):
    if data_path == None:
        data_path = '/home/jithin/datasets/imageCaptioning/flicker8k/preprocessed/'
        data_path = Path(data_path)

    assert split in ['train', 'val', 'test']

    features_fname = 'flickr8k_features_%s'%(split)
    sentences_fname = 'sentence_list_%s_flickr8k'%(split)
    lang_fname = 'lang_%s_flickr8k'%(split)

    features = load(open(data_path/features_fname, 'rb'))
    sentence_list = load(open(data_path/sentences_fname, 'rb'))
    lang = load(open(data_path/lang_fname, 'rb'))

    return features, sentence_list, lang

def collate_fn(data):
    data_new = list()
    for img_id, img, sents in data:
        for sent in sents:
            data_new.append((img_id, img, sent))
    data = data_new
    data.sort(key=lambda x: len(x[2]), reverse=True)
    img_ids, imgs, sents = zip(*data)

    # Stack images
    imgs = torch.stack(imgs, 0)

    # merge captions
    caption_lengths = [ sent.size(0) for sent in   sents]

    # zero-matrix num_captions x caption_max_length
    padded_captions = torch.zeros(len(sents), max(caption_lengths)).long()

    # fill the zero-matrix with captions. the remaining zeros are padding
    for ix, sent in enumerate(sents):
        end = caption_lengths[ix]
        padded_captions[ix, :end] = sent[:end]

    return img_ids, imgs, padded_captions, caption_lengths

if __name__ == '__main__':
    print('Running test for preprocess...')
    try:
        features, sents, lang = get_preprocessed_data('test')
        assert(len(sents) == 5000 and len(features) == 1000)
        print('OK')
    except:
        print('Failed test')

