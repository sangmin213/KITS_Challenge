from data import KidneyDataset, transform
from unet import Unet
from utils import dice_loss

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from torchvision.utils import save_image

from tqdm import tqdm
import cv2
import numpy as np

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(),lr=1e-4, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    epochs = 15
    train_len = 200 # train data 개수

    for epoch in range(epochs):
        print("-"*50)
        print("Epochs {}/{}".format(epoch+1, epochs))
        model.train()

        epoch_loss = 0.0
        total_data_len = 0

        for idx in range(train_len):
            path = str(idx).zfill(3)

            dataset = KidneyDataset(index=path,preprocess="clahe",transform=transform)
            trainloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2) # (16,1,256,256)

            data_loss = 0.0
            
            for batch in tqdm(trainloader):
                input = batch[0].to(dtype=torch.float32, device=device)
                label = batch[1].to(dtype=torch.float32, device=device)

                optimizer.zero_grad()

                pred = model(input).to(dtype=torch.float32)

                out = (pred > 0.5).float() # tensor([0.,0.,1.,1.])

                dice = dice_loss(out, label)
                loss = criterion(pred, label) + (dice if dice>0 else 0)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                data_loss += loss
                epoch_loss += loss

            print("{}/{} data loss: {}".format(idx+1, train_len, data_loss/len(dataset)))
            total_data_len += len(dataset)

        print("{}/{} epoch loss: {}".format(epoch+1, epochs, epoch_loss/total_data_len))

    # model save
    torch.save(model.state_dict(), "./model.pt")

    # optimizer save
    torch.save(optimizer.state_dict(), "./optimizer.pt")


def eval(path_idx, idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)

    checkpoint = torch.load('./model.pt')
    model.load_state_dict(checkpoint)
    model.eval()

    path = str(path_idx).zfill(3)
    dataset = KidneyDataset(index=path,preprocess="clahe",transform=transform)
    data = dataset.__getitem__(idx)
    
    input = data[0].reshape(1,1,256,256).to(dtype=torch.float32, device=device)
    label = data[1].reshape(1,1,256,256).to(dtype=torch.float32, device=device)

    pred = model(input).to(dtype=torch.float32)
    out = (pred > 0.5).float() # tensor([0.,0.,1.,1.])
    
    # # Show inference image
    # # cv2.imwrite("./input.png",np.array(input[0,0,:,:].cpu()))
    # save_image(label*255, f'label{idx}.png')
    # save_image(out*255, f'pred{idx}.png')

    # print("{}-th Eval DICE coeff: {:.3f}".format(idx, 1-dice_loss(out, label))) # dice_loss 값이 음수라면 epsilon에 의한 것. 약 loss=0 으로 추정됨

    return 1-dice_loss(out, label)

    '''
    # Compute Avg Dice score
    dataloader = DataLoader(dataset, batch_size=1)

    dice_score = 0.0
    ex = 0
    print("Evaluation...")
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        src = data[0].reshape(1,1,256,256).to(dtype=torch.float32, device=device)
        trg = data[1].reshape(1,1,256,256).to(dtype=torch.float32, device=device)

        try:
            y_hat = model(src).to(dtype=torch.float32)
            # print("Eval DICE coeff: {:.3f}".format(1-dice_loss(pred, label)))

            dice_score += (1-dice_loss(y_hat, trg)) * src.shape[0] # 평균값으로 출력된 dice_loss에 batch 크기를 곱해 총 dice_loss를 얻음

        except:
            ex += 1

    dice_score /= len(dataset)-ex

    print("exception:",ex)

    print("Avg Eval DICE coeff: {:.8f}".format(dice_score))
    '''


def main():
    # train()    
    
    dice_score = 0.0
    data_cnt = 0
    for i in tqdm(range(230,270)):
        path = str(i).zfill(3)
        dataset = KidneyDataset(index=path,preprocess="clahe",transform=transform)
        for idx in range(len(dataset)):
            dice_score += eval(i,idx)
        data_cnt += len(dataset)
    dice_score /= data_cnt
    print(dice_score)

    # print(eval(0,100))

if __name__ == "__main__":
    main()
