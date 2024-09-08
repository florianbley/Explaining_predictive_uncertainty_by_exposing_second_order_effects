import torch
import numpy

"""
This script defines the kde generative model and its training procedure.
I am not the author of this script and all credits go to Jacob Kauffmann, PhD.
See also: DOI: 10.1109/TNNLS.2022.3185901
"""

class KDE(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(2)
        super().__init__()
        self.scale = None
        self.svs = None

    # not sure if ever needed
    def forward(self, X):
        N, D = self.svs.shape
        distances = torch.cdist(X, self.svs)**2
        log_p = torch.logsumexp(-self.scale*distances, dim=-1) \
            - numpy.log(N) \
            + (D/2)*torch.log(self.scale / numpy.pi)
        return log_p

    # leave-one-out negative log likelihood
    # (assuming distances is the LOO-distance matrix)
    def LOONLL(self, distances):
        N, D = self.svs.shape
        log_p = torch.logsumexp(-self.scale*distances, dim=-1) \
            - numpy.log(N-1) \
            + (D/2)*torch.log(self.scale / numpy.pi)
        return log_p.sum()

    def fit(self, X, eps=1e-4, max_iter=50):
        # this implements the fixed-point iteration in Eq. (3)
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.422.8638&rep=rep1&type=pdf
        self.svs = X.clone()
        N, D = self.svs.shape
        distances = torch.cdist(self.svs, self.svs)**2
        distances = distances[~torch.eye(N).bool()].reshape(N,N-1)

        # rough initialization
        self.scale = 1./distances.mean()

        old_loss = float('inf')
        loss = -self.LOONLL(distances)
        n_iter = 0
        while abs(old_loss - loss) > eps and n_iter < max_iter:
            w = -self.scale*distances
            w = w - torch.logsumexp(w, dim=-1, keepdims=True)
            w = w.exp()
            sigma_squared = 1/(N*D) * (w * distances).sum()
            self.scale = 1/(2*sigma_squared)
            old_loss = loss
            loss = -self.LOONLL(distances)
            n_iter += 1
        print("Done")

    def conditional_sample(self, x, mask=None, with_noise=True, n_samples=1):
        if mask is None:
            mask = torch.zeros_like(x).bool()
        N, D = self.svs.shape

        # conditional weights
        w = -self.scale*torch.cdist(x[mask].reshape(1,-1).double(), self.svs[:,mask].double())[0]**2
        w = w.double() - torch.logsumexp(w.double(), dim=0)
        w = w.exp()

        X = x.repeat((n_samples,1)).double()
        for i in range(n_samples):
            # sample the component
            j = w.multinomial(1).item()
            # j = numpy.random.choice(N, p=w)

            # sample free pixels
            X[i,~mask] = self.svs[j,~mask].double()
            if with_noise:
                X[i,~mask] += torch.randn((~mask).sum()) * (2*self.scale)**-.5
                # X[i,~mask] += torch.from_numpy(numpy.random.normal(scale=(2*self.scale)**-.5, size=(~mask).sum())).float()

        return X.data

    def conditional_expectation(self, x, mask=None):
        if mask is None:
            mask = torch.zeros_like(x).bool()
        N, D = self.svs.shape
        x_ = x.clone()

        # conditional weights
        w = -self.scale*torch.cdist(x[mask].reshape(1,-1).double(), self.svs[:,mask].double())[0]**2
        w = w - torch.logsumexp(w, dim=0)
        w = w.exp()

        return w @ self.svs.double()
