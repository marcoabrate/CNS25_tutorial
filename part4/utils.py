import numpy as np
import torch

def create_multiple_subsampling(data, stride, is_velocity=False):
    new_length = data.shape[0]//stride if not is_velocity else data.shape[0]//stride-1
    data_multisubs = np.zeros(
        (stride, new_length, data.shape[1]),
        dtype=np.float32
    )
    for start_idx in range(stride):
        if is_velocity:
            if start_idx < stride-1:
                data_multisubs[start_idx] = data[start_idx+1:start_idx-stride+1].reshape(
                    new_length, stride, -1
                ).sum(axis=1)
            else:
                data_multisubs[start_idx] = data[start_idx+1:].reshape(
                    new_length, stride, -1
                ).sum(axis=1)
        else:
            data_multisubs[start_idx] = data[start_idx::stride]
    return data_multisubs

class SensoryDataset(torch.utils.data.Dataset):
    def __init__(self, embs, vels, rot_vels, pos, hds, tsteps=9):
        '''
        The initialisation function for the SensoryDataset class.
        At initialisation, arrays are converted to tensors.
        N is the number of trials, T is the number of time steps, and D is the number of features.

        Args:
            embs: The sensory embeddings of shape (N, T, D)
            vels: The velocity of shape (N, T-1, 2)
            rot_vels: The rotational velocities of shape (N, T-1, 1)
            pos: The x-y positions of shape (N, T, 2)
            hds: The heading directions of shape (N, T, 1)
            tsteps: The number of time steps for each batch.
                By default, this is set to 9 (seconds if frequency is 1 Hz) 
        '''
        self.embs = torch.from_numpy(embs)
        self.vels = torch.from_numpy(vels)
        self.rot_vels = torch.from_numpy(rot_vels)
        self.pos = torch.from_numpy(pos)
        self.hds = torch.from_numpy(hds)
        
        self.tsteps = tsteps
    
    def __len__(self):
        #  how many samples are in the dataset?
        return self.embs.shape[1] // self.tsteps - 1
    
    def __getitem__(self, idx):
        '''
        Returns a batch of sensory embeddings, motion signals,
        trajectory data, future sensory embeddings.
        '''
        vels, rot_vels, pos, hds, embs_labels = [], [], [], [], []

        # get a sequence of data 
        start_idx, end_idx = idx*self.tsteps, (idx + 1)*self.tsteps

        embs = self.embs[:, start_idx:end_idx]
        vels = self.vels[:, start_idx:end_idx]
        rot_vels = self.rot_vels[:, start_idx:end_idx]
        pos = self.pos[:, start_idx:end_idx]
        hds = self.hds[:, start_idx:end_idx]

        embs_labels = self.embs[:, start_idx+1 : end_idx+1]
        
        return embs, vels, rot_vels, pos, hds, embs_labels

class RNNCell(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, bias):
        super(RNNCell, self).__init__()

        self.in2hidden = torch.nn.Linear(n_inputs, n_hidden, bias=bias)
        self.hidden2hidden = torch.nn.Linear(n_hidden, n_hidden, bias=bias)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, hidden):
        igates = self.in2hidden(x)
        hgates = self.hidden2hidden(hidden)
        return self.sigmoid(igates + hgates)

class RNNModule(torch.nn.Module):
    def __init__(
        self, device, n_inputs, n_hidden, bias
    ):
        super(RNNModule, self).__init__()

        self.rnn_cell = RNNCell(n_inputs, n_hidden, bias)
        self.n_hidden = n_hidden

        self.device = device

    def forward(self, inputs, hidden=None):
        '''
        inputs is a sequence of shape (BATCH_SIZE, TIMESTEPS, N_FEATURES)
        hidden is the hidden state from the previous batch (if present), of shape (BATCH_SIZE, N_HIDDEN)

        hidden_new is a sequence of shape (BATCH_SIZE, TIMESTEPS, N_HIDDEN)
        '''
        hidden_new = torch.zeros(inputs.shape[0], inputs.shape[1], self.n_hidden).to(self.device)

        if hidden is None:
            # initialize hidden state to zero if not provided
            h_out = torch.zeros(inputs.shape[0], self.n_hidden).to(self.device)
        else:
            h_out = hidden

        window_size = inputs.shape[1]

        # loop over the sequence "inputs"
        for t in range(window_size):
            x_t = inputs[:,t,...]
            h_out = self.rnn_cell(x_t, h_out)
            hidden_new[:,t,...] = h_out

        # return all hidden states
        return hidden_new
    
class PredictiveRNN(torch.nn.Module):
    def __init__(self,
        device, n_inputs, n_hidden, n_outputs, bias=False
    ):
        super().__init__()

        self.rnn = RNNModule(
            device, n_inputs, n_hidden, bias=bias
        )

        # define the output layer
        self.hidden2outputs = torch.nn.Linear(n_hidden, n_outputs, bias=bias)

    def inputs2hidden(self, inputs, hidden):
        # just makes sure to pass the right shape of
        # the hidden state, if given
        if hidden is not None:
            return self.rnn(inputs, hidden[None, ...])
        else:
            return self.rnn(inputs)
    
    def forward(self, inputs, hidden=None):
        '''
        inputs is a sequence of shape (BATCH_SIZE, TIMESTEPS, N_FEATURES)
        hidden is the hidden state from the previous batch (if present), of shape (BATCH_SIZE, N_HIDDEN)

        hidden_new is a sequence of shape (BATCH_SIZE, TIMESTEPS, N_HIDDEN)

        outputs is a sequence of shape (BATCH_SIZE, TIMESTEPS, N_OUTPUTS)
        '''
        hidden_new = self.inputs2hidden(inputs, hidden)

        outputs = self.hidden2outputs(hidden_new)

        return outputs, hidden_new[:,-1,:]


def evaluate_rnn(
    device,
    rnn,
    dataloader,
    loss_fn,
    for_ratemaps
):
    '''
    Test the RNN for one epoch on the given dataloader.
    Args:
        device: the device to use for computation
        rnn: the RNN model to test
        dataloader: the dataloader containing the test data
        loss_fn: the loss function to use for evaluation
        for_ratemaps: ...
    Returns:
        d: a dictionary containing the test results, including losses, hidden states, positions,
            head directions, angles, outputs, and output labels
    '''
    rnn.eval()

    dictionary = {
        'batch_losses': [],
        'batch_io_dists': [],
        'hidden_states':[],
        'positions':[],
        'head_directions':[],
        'outputs':[],
        'embs_labels':[]
    }

    with torch.no_grad():
        hidden_state = None # Initialize hidden state to zeros
        for batch in dataloader:
            embs, vels, rot_vels, pos, hds, embs_labels = batch
            
            inputs = torch.cat((
                embs.squeeze(dim=0).to(device),
                vels.squeeze(dim=0).to(device),
                rot_vels.squeeze(dim=0).to(device)
            ), dim=-1)

            hidden_all_new = rnn.inputs2hidden(inputs, hidden_state)
            hidden_state = hidden_all_new[:, -1, :].detach()
            
            outputs = rnn.hidden2outputs(hidden_all_new)

            embs_labels = embs_labels.squeeze(dim=0).to(device)
            batch_losses = loss_fn(outputs, embs_labels)

            dictionary['batch_losses'].append(batch_losses.detach().item())
            dictionary['batch_io_dists'].append(loss_fn(outputs, embs.to(device)).detach().item())
            
            if for_ratemaps:
                dictionary['hidden_states'].append(hidden_all_new.detach().cpu().numpy()) # (fps, hidden_dim)
                dictionary['positions'].append(pos.squeeze(dim=0).detach().cpu().numpy()) # (fps, 2)
                dictionary['head_directions'].append(hds.squeeze(dim=0).detach().cpu().numpy()) # (fps, 2)
                dictionary['outputs'].append(outputs.detach().cpu().numpy()) # (fps, emb_dim)
                dictionary['embs_labels'].append(embs_labels.detach().cpu().numpy())
    if for_ratemaps:
        for k in ['hidden_states', 'positions', 'head_directions', 'outputs', 'embs_labels']:
            dictionary[k] = np.concatenate(dictionary[k], axis=1)

    return dictionary
