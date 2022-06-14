from utils import *
from model import *
from config import *

if __name__ == '__main__':
    text = '每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。'
    _, word2id = get_vocab()
    input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]])
    mask = torch.tensor([[1] * len(text)]).bool()

    model = torch.load(MODEL_DIR + 'model_5.pth')
    y_pred = model(input, mask)
    id2label, _ = get_label()
    label = [id2label[l] for l in y_pred[0]]
    print(text)
    print(label)

    info = extract(label, text)
    print(info)