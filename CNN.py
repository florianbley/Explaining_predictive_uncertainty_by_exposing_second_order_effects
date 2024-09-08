import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from zennit.composites import NameMapComposite
from zennit.rules import Gamma, Pass
from datasets import EPEX_FR, SeoulBike
import copy
import sklearn
from sklearn.linear_model import MultiTaskLassoCV, LinearRegression

#current dir
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# go one directory up to import utils
os.chdir("..")
# change back to original working directory
os.chdir(dir_path)


class CNN(torch.nn.Module):
    def __init__(self, n_channels, n_input_len, n_outputs, dropout_rate=0, bias=True):
        super(CNN, self).__init__()
        self.n_channels = n_channels
        self.n_input_len = n_input_len
        self.n_outputs = n_outputs
        self.main = nn.Sequential()
        self.seed = None

        layers = [
            ("conv0", nn.Conv1d(n_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=bias)),
            ("relu0", nn.ReLU()),
            ("do0", nn.Dropout(p=dropout_rate)),
            ("conv1", nn.Conv1d(16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=bias)),
            ("relu1", nn.ReLU()),
            ("do1", nn.Dropout(p=dropout_rate)),
            ("conv2", nn.Conv1d(8, out_channels=4, kernel_size=3, stride=1, padding=1, bias=bias)),
            ("relu2", nn.ReLU()),
            ("do2", nn.Dropout(p=dropout_rate)),
            ("flatten", nn.Flatten()),
            ("lin3", nn.Linear(4*self.n_input_len, 100, bias=bias)),
            ("relu3", nn.ReLU()),
            ("do3", nn.Dropout(p=dropout_rate)),
            ("lin4", nn.Linear(100, 100, bias=bias)),
            ("relu4", nn.ReLU()),
            ("lin5", nn.Linear(100, self.n_outputs, bias=bias))
        ]

        for name, layer in layers:
            self.main.add_module(name, layer)

    def forward(self, input, output_dim=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        x = input

        # forward the input through all layers
        # apply dropout after each relu except the last one
        x = self.main.conv0(x)
        x = self.main.relu0(x)
        x = self.main.do0(x)
        x = self.main.conv1(x)
        x = self.main.relu1(x)
        x = self.main.do1(x)
        x = self.main.conv2(x)
        x = self.main.relu2(x)
        x = self.main.do2(x)
        x = self.main.flatten(x)
        x = self.main.lin3(x)
        x = self.main.relu3(x)
        x = self.main.do3(x)
        x = self.main.lin4(x)
        x = self.main.relu4(x)
        x = self.main.lin5(x)

        preds = x

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
            (['conv0'], Gamma(gamma=gamma*2)),
            (['relu0'], Pass()),
            (["do0"], Pass()),
            (['conv1'], Gamma(gamma=gamma*2)),
            (['relu1'], Pass()),
            (["do1"], Pass()),
            (['conv2'], Gamma(gamma=gamma*2)),
            (['relu2'], Pass()),
            (["do2"], Pass()),
            #(['flatten'], Pass()),
            (['lin3'], Gamma(gamma=gamma)),
            (['relu3'], Pass()),
            (["do3"], Pass()),
            (['lin4'], Gamma(gamma=gamma)),
            (['relu4'], Pass()),
            (['lin5'], Gamma(gamma=0)),
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
                    output = modified_model(input)
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


if __name__ == "__main__":

    #X_train, X_test, X_val, y_train, y_test, y_val = EPEX_FR.serve_dataset(48, "channel")

    X_train, X_test, X_val, y_train, y_test, y_val = SeoulBike.serve_dataset(10, "channel")

    # make tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)


    model = CNN(n_channels=X_train.shape[1], n_input_len=X_train.shape[2], n_outputs=24)

    input_shape = X_train.shape[1]
    n_epochs = 100
    n_neurons = 300
    n_outputs = y_train.shape[1]
    dropout_rate = 0.1
    learning_rate = 0.0001
    batch_size = 128
    weight_decay = 0

    loss_fn = torch.nn.MSELoss()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_val = X_val.cuda()
        y_val = y_val.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = np.inf
    for epoch in range(n_epochs):
        # make a list of batch indices
        train_inds = list(range(len(X_train)))
        # shuffle the list
        np.random.shuffle(train_inds)
        # split the list into batches

        n_batches = int(len(X_train) / batch_size)
        batch_ind_list = [list(train_inds[i * batch_size: (i + 1) * batch_size]) for i in range(n_batches)]

        model.train()
        for batch_inds in batch_ind_list:
            X_batch, y_batch = X_train[batch_inds], y_train[batch_inds]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_loss = loss_fn(y_pred_train, y_train)

            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, y_val)

        print("Epoch: {}, Training loss: {}, Validation loss: {}".format(epoch, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            n_not_improved = 0
        else:
            n_not_improved += 1
            if n_not_improved > 10:
                break

    test_mse_cnn = loss_fn(best_model(X_test), y_test)

    #X_train, X_test, X_val, y_train, y_test, y_val = EPEX_FR.serve_dataset(48, "tabular")
    X_train, X_test, X_val, y_train, y_test, y_val = SeoulBike.serve_dataset(10, "tabular")


    # use linear model, run some hp optimization on the validation set
    lin_model = LinearRegression()
    lin_model.fit(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]))
    test_preds = lin_model.predict(X_test)
    test_acc_linear = loss_fn(torch.tensor(test_preds, dtype=torch.float32), torch.Tensor(y_test))


    print("Done")