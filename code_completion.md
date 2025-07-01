### PART 1

```python
# YOUR CODE HERE (1)
env = Environment(params={'scale': ENV_DIM-2*ENV_EPS, 'aspect':1})
```

```python
# YOUR CODE HERE (2)
agent = Agent(env, params=agent_params)
agent.velocity /= 1e3 # slow down the agent when it first starts

for i in range(int(LENGTH/DT)):
    agent.update(dt=DT)

positions = agent.history['pos']
velocities = agent.history['vel']
```

```python
# YOUR CODE HERE (3)
r = Renderer(
  blender_exec='/Applications/Blender.app/Contents/MacOS/Blender',
  config={
      'env_file': 'environment/box_messy.blend',
  }
)
r.render(positions, head_directions)
```


### PART 2

```python
# YOUR CODE HERE (1)
KERNEL_SIZES = [(3,3), (4,4), (4,4), (3,3)] # kernel sizes for the convolutional layers
KERNEL_STRIDES = [1, 2, 2, 1] # strides
CHANNELS = [8, 8, 16, 16] # number of channels

EMBEDDING_DIM = 100 # the number of neurons in the hidden state (aka embedding dimension)
```

```python
# YOUR CODE HERE (2)
# forward and backward passes
batch_recon = dec(enc(batch))
loss = loss_fn(batch_recon, batch)
loss.backward()
optimizer.step()
```

```python
# YOUR CODE HERE (3)
# forward pass
batch_recon = dec(enc(batch))
loss = loss_fn(batch_recon, batch)
```


### PART 3

```python
# YOUR CODE HERE (1)
#  how many samples are in the dataset?
return self.embs.shape[1] // self.tsteps - 1
```

```python
# YOUR CODE HERE (2)
# get a sequence of data 
start_idx, end_idx = idx*self.tsteps, (idx + 1)*self.tsteps

embs = self.embs[:, start_idx:end_idx]
vels = self.vels[:, start_idx:end_idx]
rot_vels = self.rot_vels[:, start_idx:end_idx]
pos = self.pos[:, start_idx:end_idx]
hds = self.hds[:, start_idx:end_idx]

embs_labels = self.embs[:, start_idx+1 : end_idx+1]
```

```python
# YOUR CODE HERE (3)
# Define the RNN cell class
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
```

```python
# YOUR CODE HERE (4)
# loop over the sequence "inputs"
for t in range(window_size):
    x_t = inputs[:,t,...]
    h_out = self.rnn_cell(x_t, h_out)
    hidden_new[:,t,...] = h_out
```

```python
# YOUR CODE HERE (5)
# define the output layer
self.hidden2outputs = torch.nn.Linear(n_hidden, n_outputs, bias=bias)
```

```python
# YOUR CODE HERE (6)
# concatenate the embeddings, velocities, and rotational velocities
# and pass the inputs through the RNN
inputs = torch.cat((
    embs.squeeze(dim=0).to(DEVICE),
    vels.squeeze(dim=0).to(DEVICE),
    rot_vels.squeeze(dim=0).to(DEVICE)
), dim=-1)

outputs, hidden_new = rnn(inputs, hidden_state)
```

```python
# YOUR CODE HERE (7)
# compute the loss and its gradients
loss = loss_fn(outputs, embs_labels)
loss.backward()

# (optional) clip the gradients
# torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)

optimizer.step()

# Assign new RNN hidden state to variable.
# Detach it to prevent backpropagation
# through the entire history
hidden_state = hidden_new.detach()
```

### PART 4

```python
# YOUR CODE HERE (1)
# calculate the activations for each cell, for each binned position
activations = bin_data(pos, n_bins=n_bins, limits=limits, weights=hidden_states[:,i])


# YOUR CODE HERE (2)
# obtain the rate map for each cell by dividing the activations by the occuopancy
rate_maps[i,:,:] = np.divide(
    activations, occupancy, where=occupancy!=0,
    out=np.nan*np.ones_like(activations)
)

# YOUR CODE HERE (3)
# calculate the activations for each cell, for each binned head direction
activations = bin_data(
    head_directions, n_bins=n_bins, limits=limits, weights=hidden_states[:,i]
)

# YOUR CODE HERE (4)
# obtain the polar map for each cell by dividing the activations by the occuop
polar_maps[i] = np.divide(
    activations, occupancy, where=occupancy!=0,
    out=np.nan*np.ones_like(activations)
)

# YOUR CODE HERE (5)
# use the equation above, calculate the SI (assuming we are measuring in bits per second)
si = np.sum(
    _occ_prob[mask] * _rate[mask] * np.log2(_rate[mask] / _rate_mean)
)

si /= _rate_mean  # convert to bits per spike

# YOUR CODE HERE (6)
# calculate resultant vector as sum of cos & sin angles
rv = np.sum(polar_map * np.exp(1j * bins))

```