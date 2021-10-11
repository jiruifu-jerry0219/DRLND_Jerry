""""The class FCDP stands for fully-connected discrete action Policy
The network allows you to do stochastic policy for discrete action""""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCDP(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                hidden_dims = (32, 32),
                init_std = 1,
                activation_fc = F.relu):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(
            input_dim, hidden_dims[0]
        )
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layers = nn.Linear(
                hidden_dims[i], hidden_dims[i+1]
            )
            self.hidden_layers.append(hidden_layers)

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state):
        x = state_size
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype = torch.float32)
            x = x.unsqueeze(0)

        x = self.activation_fc(self.input_layer(x))

        for hidden_layers in self.hidden_layers:
            x = self.activation_fc(hidden_layers(x))

        return self.output_layer(x)

    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
