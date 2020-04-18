import json
from collections import namedtuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

annotations = namedtuple('Annotations',['image_id','sentences'])

class Flickr8k(Dataset):
    """ for flickr 8k dataset."""

    def __init__(self, img_dir, ann_file, vocab, split='train', transform=None, target_transform=None):
        """
        Args:
            root (str): The root dir that points to the Flickr images.
            ann_file (str): The file that contains the annotations for the images.
            split ['train', 'val', 'test']: This decides which partition to load.
            transform: Transforms for image.
            target_transforms: transforms for sentences.
        """
        self.img_dir = Path(img_dir)
        self.vocab = vocab
        assert split in ['train', 'test', 'val']
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = list()

        # indices when spliting the json file
        if self.split == 'train':
            m, n = 0, 6000
        elif self.split == 'val':
            m, n = 6000, 7000
        elif self.split == 'test':
            m, n = 7000, 8000

        with open(ann_file, 'r') as ann_file:
            ann_json = json.load(ann_file)
            for image in ann_json['images'][m : n]:
                image_id = image['filename']
                sentences_list = list()
                for sentence in image['sentences']:
                    sentences_list.append(sentence['tokens'])
                self.annotations.append(annotations(image_id, sentences_list))

                assert image['split'] == self.split

            print('loading %s complete'%(self.split))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations[index].image_id

        img = Image.open(self.img_dir/img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[index].sentences
        captions = [torch.tensor(
                    [self.vocab.SOS_token] +
                    [self.vocab(word) for word in sent]  +
                    [self.vocab.EOS_token]) for sent in target]

        return img_id, img, captions
