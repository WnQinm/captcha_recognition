# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN

# Hyper Parameters
num_epochs = 30
learning_rate = 0.0002

def main():
    cnn = CNN()
    cnn.train()
    cnn.load_state_dict(torch.load("./model.pt"))
    cnn = cnn.to(torch.device('cuda'))
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_predict_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

            images = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.detach().item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pt")
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.detach().item())
    torch.save(cnn.state_dict(), "./model.pt")
    print("save last model")

if __name__ == '__main__':
    main()


