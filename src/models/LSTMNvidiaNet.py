import torch.nn as nn
import torch

__all__ = ['LSTMNvidiaNet']

class LSTMNvidiaNet(nn.Module):

    def __init__(self, num_blendshapes=51, num_emotions=16):
        super(LSTMNvidiaNet, self).__init__()

        self.num_blendshapes = num_blendshapes
        self.num_emotions = num_emotions

        # emotion network with LSTM
        self.emotion = nn.LSTM(input_size=32, hidden_size=128, num_layers=1,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(128*2, 150),
            nn.ReLU(),
            nn.Linear(150, self.num_emotions)
        )


        # formant analysis network
        self.formant = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )

        # articulation network
        self.conv1 = nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.conv2 = nn.Conv2d(256+self.num_emotions, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.conv5 = nn.Conv2d(256+self.num_emotions, 256, kernel_size=(4,1), stride=(4,1))
        self.relu = nn.ReLU()

        # output network
        self.output = nn.Sequential(
            nn.Linear(256+self.num_emotions, 150),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(150, self.num_blendshapes)
        )

    def forward(self, x):
        # extract emotion state
        e_state, _ = self.emotion(x[:, ::2]) # input features are 2* overlapping
        e_state = self.dense(e_state[:, -1, :]) # last
        e_state = e_state.view(-1, self.num_emotions, 1, 1)

        x = torch.unsqueeze(x, dim=1)
        # convolution
        x = self.formant(x)

        # conv+concat
        x = self.relu(self.conv1(x))
        x = torch.cat((x, e_state.repeat(1, 1, 32, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 16, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 8, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 4, 1)), 1)

        x = self.relu(self.conv5(x))
        x = torch.cat((x, e_state), 1)

        # fully connected
        x = x.view(-1, num_flat_features(x))
        x = self.output(x)

        return x

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
