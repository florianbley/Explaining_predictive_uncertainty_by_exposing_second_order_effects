"""
Example script which demonstrates how to compute CovLRP uncertainty explanations.
"""

from datasets import kin8nm
import torch
import numpy as np
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma, Pass
from zennit.types import Linear, Activation


def numpy_to_torch(tup):
    return tuple(torch.tensor(arr, dtype=torch.float32) for arr in tup)


def train(model, X_train, y_train, X_val, y_val):
    """
    Simple training loop for the sequential neural network model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    old_val_loss = np.inf
    best_model = None
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = loss_fn(model(X_val), y_val)
        if val_loss < old_val_loss:
            best_model = model
            old_val_loss = val_loss
        print("Epoch: {}, Training loss: {}, Validation loss: {}".format(epoch, loss, val_loss))
    return best_model


def lrp(seq_model, X, gamma=0.3):
    """Simple Zennit LRP implementation using gamma for sequential models."""
    layer_map = [
        (Activation, Pass()),  # ignore activations
        (Linear, Gamma(gamma=gamma))  # this is the dense Linear, not any
    ]
    composite_gamma = LayerMapComposite(layer_map=layer_map)
    input = X.clone().detach().requires_grad_(True)
    with composite_gamma.context(seq_model) as modified_model:
        output = modified_model(input)
        attribution, = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))
    return attribution


if __name__ == '__main__':
    # Load the kin8nm dataset
    X_train, X_test, X_val, y_train, y_test, y_val = numpy_to_torch(kin8nm.serve_dataset())

    model_list = []
    # initialise 10 models randomly and add them to the list
    for i in range(10):
        # fix different init seeds
        torch.manual_seed(i)
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, y_train.shape[1])
        )
        train(model, X_train, y_train, X_val, y_val)
        model_list.append(model)

    # compute uncertainty estimates on the test set
    test_preds = torch.stack([model(X_test) for model in model_list])
    test_predictive_uncertainty = test_preds.var(dim=0).mean(dim=1)

    # Now compute the CovLRP uncertainty explanation
    # First we compute LRP explanations for each model prediction
    model_explanations = torch.stack([lrp(model, X_test, gamma=0.3) for model in model_list])

    # Iterate over the data points and compute covariance matrix for each data point heatmap
    Cov_LRP = []
    for n in range(model_explanations.shape[1]):
        lrps_n = model_explanations[:, n, :]
        Cov_LRP.append(torch.cov(lrps_n.T, correction=0))
    Cov_LRP = torch.stack(Cov_LRP)

    Cov_LRP_diag = Cov_LRP.diagonal(dim1=-2, dim2=-1)
    Cov_LRP_marg = Cov_LRP.sum(dim=1)

    print("Demo is finished.")


