"""
Example script which demonstrates how to compute CovLRP uncertainty explanations.
"""

from datasets import wineQuality
import torch
import numpy as np
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma, Pass
from zennit.types import Linear, Activation
import matplotlib.pyplot as plt


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

def plot_results(X_test, feature_names, Cov_LRP_diag):
    fig, ax = plt.subplots(3, 4, figsize=(13, 8))
    # iterate over all subplots and plot a scatter of feature values and Cov_LRP_diag values
    for i in range(3):
        for j in range(4):
            if j == 0:
                # set y axis label to "Predictive variance"
                ax[i, 0].set_ylabel("Uncertainty Relevance")
            feature_ind = i * 4 + j
            # keep one empty
            if feature_ind == 11:
                continue
            ax[i, j].scatter(X_test[:, feature_ind].detach().numpy(), Cov_LRP_diag[:, feature_ind].detach().numpy())
            ax[i, j].set_title(feature_names[feature_ind] + "\n (standardized)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load the Wine Quality dataset
    X_train, X_test, X_val, y_train, y_test, y_val = numpy_to_torch(wineQuality.serve_dataset())
    feature_names = wineQuality.get_feature_names()
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
    N_test = len(X_test)
    Cov_LRP = []
    for n in range(N_test):
        lrps_n = model_explanations[:, n, :] # heatmaps for data point n
        lrp_cov_n = torch.cov(lrps_n.T, correction=0) # covariance heatmap for data point n
        Cov_LRP.append(lrp_cov_n)
    Cov_LRP = torch.stack(Cov_LRP) # covariance type uncertainty explanation

    # as proposed in the main paper, we can simplify the covariance matrix to the diagonal and marginal
    Cov_LRP_diag = Cov_LRP.diagonal(dim1=-2, dim2=-1)
    Cov_LRP_marg = Cov_LRP.sum(dim=1)

    # make a scatter plot for uncertainty relevance vs feature values
    plot_results(X_test, feature_names, Cov_LRP_diag)

    print("Demo is finished.")


