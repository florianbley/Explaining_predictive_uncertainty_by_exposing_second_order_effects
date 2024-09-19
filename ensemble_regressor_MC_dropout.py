"""
Ensemble-type implementation of MC-dropout. For the sake of reproducibility we implement MC-Dropout uncertainty estimation
as an ensemble of model clones with different forward seeds for dropout.
"""


import numpy as np
import torch
import copy


class Ensemble_regressor_MC_dropout():
    def __init__(self, model, MC_passes=10):
        self.models = []

        for m in range(MC_passes):
            model_copy = copy.deepcopy(model)
            model_copy.set_internal_seed(m)
            self.models.append(model_copy)

    def forward(self, X):
        ensemble_output = torch.Tensor(X.shape[0], len(self.models), self.models[0].n_outputs)
        for m, model in enumerate(self.models):
            # set model to train mode
            model.train()
            # set seet for dropout
            torch.manual_seed(m)
            ensemble_output[:, m, :] = model.forward(X)
        return ensemble_output

    def preactivation_forward(self, X):
        return [model.preactivation_forward(X, dropout=False) for model in self.models]

    def mean_prediction(self, X):
        return torch.stack([model.forward(X, dropout=False) for model in self.models]).mean(0)

    def epistemic_variance(self, X):
        preds = self.forward(X)
        vars = preds.var(1).sum(1)
        return vars

    def move_to_cuda(self):
        for model in self.models:
            model.cuda()

    def move_to_cpu(self):
        for model in self.models:
            model.cpu()

    def eval(self):
        for model in self.models:
            model.eval()

    def lrp_all(self, explain_sample, gamma=0, R_upper=None):
        explanations = []
        for m, model in enumerate(self.models):
            torch.manual_seed(m)
            if next(model.parameters()).is_cuda:
                # move to cpu
                model = model.cpu()
            if R_upper is not None:
                R_upper_model = R_upper[:, m, :]
            else:
                R_upper_model = None
            R_model = model.lrp(explain_sample, gamma=gamma, R_upper=R_upper_model)
            explanations.append(R_model)
        return np.stack(explanations)

    def lrp(self, explain_sample, gamma=0):

        model_preds = self.forward(explain_sample)
        cond_mean = model_preds.mean(1)
        R_upper = ((model_preds - cond_mean[:, None, :]) ** 2)/(len(self.models))
        R_all = self.lrp_all(explain_sample, R_upper=R_upper, gamma=gamma)
        R = R_all.sum(0).sum(1)
        return R