from pickle import dump
from pickle import load
from pathlib import Path

from torchvision import models
from torchvision import transforms
import torch
from torch import nn
from tqdm import tqdm

from datasets import Flickr8k

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS


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

def indexesFromSentence(lang: Lang, sentence: list):
    return [lang.word2index[word] for word in sentence]

def tensorFromSentence(lang, sentence):
    device = torch.device('cpu' if torch.cuda.is_available else 'cpu')
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorForImageCaption(features_dict, sentence_tuple, lang):
    """
    This created the input_img, imput_vec pair that is used for training
    the model.

    Args:
        sentence_tuple: tuple with the img idx and tokenized sentence
        feature_dict: dict that maps the img idx and the feature tensor
        lang: The language model used.
    """
    idx, sentence = sentence_tuple
    target_tensor = tensorFromSentence(lang, sentence)
    features  = features_dict[idx]
    features = torch.from_numpy(features)
    return features, target_tensor
    return features, sentence_list,lang
