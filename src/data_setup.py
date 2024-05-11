import torch
from torch.utils.data import Dataset
from datasets import load_dataset

import numpy as np


class NMTDataset(Dataset):
    def __init__(self, cfg, split='train', prefix=''):
        super().__init__()
        self.cfg = cfg

        src_texts, tgt_texts = self.read_data(split, prefix)

        self.src_input_ids = self.text_to_sequence(src_texts)
        self.labels = self.text_to_sequence(tgt_texts)

    def read_data(self, split, prefix):
        dataset = load_dataset("mt_eng_vietnamese",
                               "iwslt2015-en-vi",
                               split=split,
                               cache_dir=self.cfg.cache_dir)

        src_texts = [sample['translation'][self.cfg.src_lang] for sample in dataset]
        tgt_texts = [sample['translation'][self.cfg.tgt_lang] for sample in dataset]

        return src_texts, tgt_texts

    def text_to_sequence(self, text):
        inputs = self.cfg.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors='pt'
        )

        return inputs.input_ids

    def __getitem__(self, idx):
        return {
            'input_ids': self.src_input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return np.shape(self.src_input_ids)[0]


class Augmented_NMTDataset(NMTDataset):
    def __init__(self, cfg, augmented_src=[], augmented_tgt=[], split='train', prefix=''):
        super().__init__(cfg, split, prefix)

        augmented_src_sequence = self.text_to_sequence(augmented_src)
        augmented_tgt_sequence = self.text_to_sequence(augmented_tgt)

        self.src_input_ids = torch.stack((self.src_input_ids, augmented_src_sequence), dim=0)
        self.labels = torch.stack((self.labels, augmented_tgt_sequence), dim=0)

    def read_data(self, split, prefix):
        return super().read_data(split, prefix)

    def text_to_sequence(self, text):
        return super().text_to_sequence()

    def __getitem__(self, idx):
        return {
            'input_ids': self.src_input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return super().__len__()