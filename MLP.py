import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from zennit.composites import NameMapComposite
from zennit.rules import Gamma, Pass

#current dir
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# go one directory up to import utils
os.chdir("..")
# change back to original working directory
os.chdir(dir_path)


class MLP(torch.nn.Module):
    def __init__(self, input_shape, n_outputs, dropout_rate=0, n_layers=1, n_neurons=300, bias=True):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.main = nn.Sequential()

        self.seed = None

        """layers = []
        for l in range(n_layers):
            n_neurons_layer = int((n_layers - l)*n_neurons)
            layers.append(("lin" + str(l), nn.Linear(input_shape, n_neurons_layer, bias=bias)))
            layers.append(("relu" + str(l), nn.ReLU()))
        layers.append(("lin" + str(n_layers), nn.Linear(n_neurons_layer, self.n_outputs, bias=bias))"""

        layers = [
            ("lin0", nn.Linear(input_shape, 3 * n_neurons, bias=bias)),
            ("relu0", nn.ReLU()),
            ("do0", nn.Dropout(p=dropout_rate)),
            ("lin1", nn.Linear(3 * n_neurons, 2 * n_neurons, bias=bias)),
            ("relu1", nn.ReLU()),
            ("do1", nn.Dropout(p=dropout_rate)),
            ("lin2", nn.Linear(2 * n_neurons, n_neurons, bias=bias)),
            ("relu2", nn.ReLU()),
            ("do2", nn.Dropout(p=dropout_rate)),
            ("lin3", nn.Linear(n_neurons, self.n_outputs, bias=bias))
        ]
        for name, layer in layers:
            self.main.add_module(name, layer)

    def forward(self, input, output_dim=None):

        if self.seed is not None:
            torch.manual_seed(self.seed)

        x = input

        x = self.main.lin0(x)
        x = self.main.relu0(x)
        x = self.main.do0(x)
        x = self.main.lin1(x)
        x = self.main.relu1(x)
        x = self.main.do1(x)
        x = self.main.lin2(x)
        x = self.main.relu2(x)
        x = self.main.do2(x)
        preds = self.main.lin3(x)

        #preds = self.main(x)
        if output_dim is not None:
            return preds[:, output_dim]
        else:
            return preds


    def get_layers(self):
        return self.main

    def set_internal_seed(self, seed):
        self.seed = seed

    def collect_activations(self, X, layers):
        activations = []
        activations.append(X)
        for layer in layers[:-1]:
            X = layer(X)
            activations.append(X)
        return activations

    def MC_predictive_uncertainy(self, X, n_samples=10):
        # if x only one dimensional, add empty one first
        original_onedim = len(X.shape) == 1
        if original_onedim:
            X = X[None, :]
        # set seed for reproducibility
        torch.manual_seed(0)
        # set training mode
        self.train()
        # produce n_samples predictions for X
        preds = []
        for i in range(n_samples):
            preds.append(self.forward(X))
        preds = torch.stack(preds)
        # predictive uncertainty is variance of predictions over n_samples
        pred_var = torch.var(preds, dim=0).sum(1)
        # if x was only one dimensional, remove empty one again
        if original_onedim:
            pred_var = pred_var[0]
        return pred_var

    def lrp(self, explain_sample, gamma=0, R_upper=None):
        name_map = [
            (['lin0'], Gamma(gamma=gamma)),
            (['relu0'], Pass()),
            (['do0'], Pass()),
            (['lin1'], Gamma(gamma=gamma)),
            (['relu1'], Pass()),
            (['do1'], Pass()),
            (['lin2'], Gamma(gamma=gamma)),
            (['relu2'], Pass()),
            (['do2'], Pass()),
            (['lin3'], Gamma(gamma=0))
        ]
        explanations = []
        for i, sample in enumerate(explain_sample):
            sample = sample[None, :]
            sample_output_dim_explanations = []
            for d in range(self.n_outputs):
                composite_name_map = NameMapComposite(name_map=name_map)
                input = sample.clone().detach().requires_grad_(True)
                with composite_name_map.context(self.main) as modified_model:
                    # execute the hooked/modified model
                    output = modified_model.forward(input)
                    # set all entries except the dth one to zero
                    output = output * (torch.arange(self.n_outputs) == d).float()[None]

                    if R_upper is not None:
                        upper_relevance = R_upper[i] * (torch.arange(self.n_outputs) == d).float()[None]
                    else:
                        upper_relevance = output
                    # compute the attribution via the gradient
                    attribution, = torch.autograd.grad(
                        output, input, grad_outputs=upper_relevance, create_graph=True, retain_graph=True
                    )
                sample_output_dim_explanations.append(attribution.detach().numpy())
            sample_output_dim_explanations = np.vstack(sample_output_dim_explanations)
            explanations.append(sample_output_dim_explanations)
        explanations = np.stack(explanations)
        return explanations


