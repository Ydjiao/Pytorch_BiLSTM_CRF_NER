from utils import *
from model import *
from config import *

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCH):
        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 10 == 0:
                print('>> epoch:', e, 'loss:', loss.item())

        torch.save(model, MODEL_DIR + f'model_{e}.pth')