import torch
from torch.utils import data
from config import *
import pandas as pd


def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)

def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)


class Dataset(data.Dataset):
    def __init__(self, type='train', base_len=50):
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_points()

    def get_points(self):
        self.points = [0]
        i = 0
        while True:
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i += 1

    def __len__(self, len):
        return len(self.points) -1

    def __getitem__(self, index):
        df = self.df[self.points[index]:self.points[index + 1]]
        word_unk_id = self.word2id[WORD_UNK]
        label_o_id = self.label2id['O']
        input = [self.word2id.get(w, word_unk_id) for w in df['word']]
        target = [self.label2id.get(l, label_o_id) for l in df['label']]
        return input, target

if __name__ == '__main__':
    dataset = Dataset()
