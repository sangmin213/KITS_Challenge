from data import KidneyDataset, transform
from unet import Unet
from utils import dice_loss

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from torchvision.utils import save_image

from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(),lr=1e-4, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    epochs = 5

    for epoch in range(epochs):
        print("Epochs {}/{}".format(epoch+1, epochs))
        model.train()

        epoch_loss = 0.0

        for idx in range(53):
            print("{}/{} data".format(idx+1,53),end="")
            path = str(idx).zfill(3)

            dataset = KidneyDataset(index=path,transform=transform)
            trainloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2) # (16,1,256,256)

            data_loss = 0.0
            
            for batch in tqdm(trainloader):
                input = batch[0].to(dtype=torch.float32, device=device)
                label = batch[1].to(dtype=torch.float32, device=device)

                optimizer.zero_grad()

                pred = model(input).to(dtype=torch.float32)

                loss = criterion(pred, label) + dice_loss(pred, label)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                data_loss += loss
                epoch_loss += loss

            print("{}/{} data loss: {}".format(idx, 53, data_loss/len(dataset)))

        print("{}/{} epoch loss: {}".format(epoch+1, epochs, epoch_loss/(len(dataset)*53)))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # model save
    torch.save(model.state_dict(), "./model.pt")


def eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)

    checkpoint = torch.load('./model.pt')
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = KidneyDataset(transform=transform)
    data = dataset.__getitem__(300)
    
    input = data[0].reshape(1,1,256,256).to(dtype=torch.float32, device=device)
    label = data[1].reshape(1,1,256,256).to(dtype=torch.float32, device=device)

    pred = model(input).to(dtype=torch.float32)
    out = (pred > 0.5).float() # tensor([0.,0.,1.,1.])

    save_image(label*255, 'label.png')
    save_image(out*255, 'pred.png')
            

def main():
    # train()    
    eval()

if __name__ == "__main__":
    main()
