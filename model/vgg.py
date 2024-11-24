import numpy as np
import torch
import torchvision.transforms
from torch import nn



class VGG16_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,(3,3),1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.layer2 =nn.Sequential(
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.flatten1 = nn.Flatten()  #真正的vgg后面应该还有三个全连接层，再softmax输出结果


    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


def main():

    '''
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicModel().to(device)
    # model = BasicModel()
    model.train()
    input_data = np.random.rand(1, 3, 8, 8).astype(np.float32)
    print("input shape", input_data.shape)
    # input_data = torch.tensor(input_data)
    input_data = torch.tensor(input_data).to(device)
    print(input_data.device)
    output_data = model(input_data)
    output_data = output_data.to('cpu').detach().numpy()
    # output_data = output_data.detach().numpy()
    print("output shape", output_data.shape)
    final_time = time.time()
    print(final_time - start_time)
'''

    x_date = np.random.rand(2,3, 224, 224).astype(np.float32)
    x = torch.from_numpy(x_date)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16_Model()
    model.eval()
    pred = model(x)
    print(pred.shape)

if __name__ == "__main__":
    main()
