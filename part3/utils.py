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
