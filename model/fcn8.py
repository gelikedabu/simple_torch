import torch.cuda
import time
from vgg import *
import torch.nn.functional as F

class FCN8_model(nn.Module):
    def __init__(self, num_classes=59):
        super().__init__()
        backbone = VGG16_Model()
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layer5 = backbone.layer5
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096 , 1 ),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.score = nn.Conv2d(4096, out_channels=num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, out_channels=num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512 , out_channels=num_classes, kernel_size=1)

        self.up_output = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes,
                                            kernel_size=4, stride=2,padding=1)

        self.up_pool4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes,
                                            kernel_size=4, stride=2,padding=1)

        self.up_final = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes,
                                            kernel_size=16, stride=8,padding=4)

    def forward(self,inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        pool3 = x
        x = self.layer4(x)
        pool4 = x
        x = self.layer5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.score(x)
        print(x.shape)
        x = self.up_output(x)
        up_output = x
        x = self.score_pool4(pool4)
        # x = x[:, :, 5:5+up_output.shape[2], 5:5+up_output.shape[3]]
        up_pool4 = x
        x = up_pool4 + up_output
        x = self.up_pool4(x)
        up_pool4 = x

        x = self.score_pool3(pool3)
        # # x = x[:, :, 9:9+up_pool4.shape[2], 9:9+up_pool4.shape[3]]
        up_pool3 = x
        x = up_pool3 + up_pool4
        x = self.up_final(x)
        # # x = x[:, :, 31:31+inputs.shape[2], 31:31+inputs.shape[3]]

        return x


def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    x_date = np.random.rand(2,3,512,512).astype(np.float32)
    x = torch.from_numpy(x_date)
    x = x.to(device)
    model = FCN8_model().to(device)
    model.eval()
    pred = model(x)
    print(pred.shape)
    print(pred.device)
    final_time = time.time()
    print(final_time - start_time)

if __name__ == "__main__":
    main()