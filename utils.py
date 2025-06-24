import torch

 
class RNNModule(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, nonlinearity, device, input_bias=True, hidden_bias=True):
        super(RNNModule, self).__init__()

        self.rnn_cell = RNNCell(n_inputs, n_hidden, nonlinearity, input_bias, hidden_bias)
        self.n_hidden = n_hidden

        self.device = device

    def forward(self, x, hidden=None):
        # x: [BATCH SIZE, TIME, N_FEATURES]
        
        output = torch.zeros(x.shape[0], x.shape[1], self.n_hidden).to(self.device)

        if hidden is None:
            h_out = torch.zeros(x.shape[0], self.n_hidden) # initialize hidden state
            h_out = h_out.to(self.device)
        else:
            h_out = hidden

        window_size = x.shape[1]

        # loop over time
        for t in range(window_size):
            x_t = x[:,t,...]
            h_out = self.rnn_cell(x_t, h_out)
            output[:,t,...] = h_out

        # return all outputs, and the last hidden state
        return output, h_out

class RNNCell(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, nonlinearity, input_bias, hidden_bias):
        super(RNNCell, self).__init__()

        if nonlinearity == 'sigmoid':
            activation_fn = torch.sigmoid
        else:
            print("[!!!] WARNING: activation function not recognized, using identity")
            activation_fn = torch.nn.Identity()

        self.in2hidden = torch.nn.Linear(n_inputs, n_hidden, bias=input_bias)
        self.hidden2hidden = torch.nn.Linear(n_hidden, n_hidden, bias=hidden_bias)

        self.activation_fn = activation_fn

    def forward(self, x, hidden):
        igates = self.in2hidden(x)
        hgates = self.hidden2hidden(hidden)
        return self.activation_fn(igates + hgates)
