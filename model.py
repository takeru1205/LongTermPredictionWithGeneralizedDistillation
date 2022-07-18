import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(model, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        h, _= self.rnn(inputs)
        output = self.fc(h[:, -1])

        return output


class teacher_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(model, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        h, _= self.rnn(inputs)
        output = nn.Sigmoid(self.fc(h[:, -1]))

        return output

