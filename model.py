import torch.nn as nn
from config import *
from torchcrf import CRF
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE)

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.viterbi_decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask).mean()

if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50))
    print(model(input, None).shape)