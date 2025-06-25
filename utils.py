import torch
import numpy as np
 
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

    d = {
        'batch_losses': [],
        'hidden_states':[],
        'positions':[],
        'head_directions':[],
        'outputs':[],
        'embs_labels':[]
    }

    with torch.no_grad():
        hidden_state = None # Initialize hidden state to zeros
        for i, batch in enumerate(dataloader):
            embs, vels, rot_vels, pos, hds, embs_labels = batch
            
            inputs = torch.cat((
                embs.squeeze(dim=0).to(device),
                vels.squeeze(dim=0).to(device),
                rot_vels.squeeze(dim=0).to(device)
            ), dim=-1)

            hidden_all_new = rnn.inputs2hidden(inputs, hidden_state)
            hidden_new = hidden_all_new[:, -1, :]
            outputs = rnn.hidden2outputs(hidden_all_new)

            embs_labels = embs_labels.squeeze(dim=0).to(device)
            batch_losses = loss_fn(outputs, embs_labels)

            d['batch_losses'].append(batch_losses.detach().item())
            
            if for_ratemaps:
                d['hidden_states'].append(hidden_all_new.detach().cpu().numpy())                           # (fps, hidden_dim)
                d['positions'].append(pos.squeeze(dim=0).detach().cpu().numpy())                # (fps, 2)
                d['head_directions'].append(hds.squeeze(dim=0).detach().cpu().numpy())                # (fps, 2)
                d['outputs'].append(outputs.detach().cpu().numpy())            # (fps, emb_dim)
                d['embs_labels'].append(embs_labels.detach().cpu().numpy())
    if for_ratemaps:
        for k in ['hidden_states', 'positions', 'head_directions', 'outputs', 'embs_labels']:
            d[k] = np.concatenate(d[k], axis=1)

    return d
