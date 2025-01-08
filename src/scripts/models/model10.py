model = """
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleRNN, self).__init__()
        
        # Fixed parameters
        self.hidden_size = 64  # Number of hidden neurons
        self.num_layers = 2    # Number of RNN layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state (h0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Take the output from the last time step
        out = out[:, -1, :]  # Get last time step's output

        # Pass through the fully connected layer
        out = self.fc(out)
        return out
"""
