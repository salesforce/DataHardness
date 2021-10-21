import numpy as np
import LinConGauss as lcg
import argparse
import torch
from scipy.linalg import sqrtm


class GaussianBayes:
    def __init__(self, means, cov, marginals=None):
        self.means = means
        self.cov = cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.K = len(means)
        if marginals is None:
            self.marginals = np.array([1./self.K]*self.K)

    def _integrate(self, A, b, n_samples=2000):
        lincon = lcg.LinearConstraints(A=A, b=b)
        subsetsim = lcg.multilevel_splitting.SubsetSimulation(linear_constraints=lincon,
                                                      n_samples=100,
                                                      domain_fraction=0.5,
                                                      n_skip=3)
        subsetsim.run(verbose=False)
        shifts = subsetsim.tracker.shift_sequence
        hdr = lcg.multilevel_splitting.HDR(linear_constraints=lincon,
                                   shift_sequence=shifts,
                                   n_samples=n_samples,
                                   n_skip=9,
                                   X_init=subsetsim.tracker.x_inits())
        hdr.run(verbose=False)
        hdr_integral = hdr.tracker.integral()
        return hdr_integral

    def _form_constraints(self, k):
        not_k_ix = [i for i in range(self.K) if i != k]
        mu_k = self.means[k]
        b = []
        A = []
        for j in not_k_ix:
            mu_j = self.means[j]
            bjk = np.dot(mu_j, np.dot(self.cov_inv, mu_j)) - np.dot(mu_k, np.dot(self.cov_inv, mu_k))
            Ajk = 2*(np.dot(self.cov_inv, mu_k) - np.dot(self.cov_inv, mu_j))
            bjk += np.dot(Ajk, mu_k)
            Ajk = np.dot(np.sqrt(self.cov), Ajk) # NOTE: this assumes covariance is diagonal
            b.append(bjk)
            A.append(Ajk)

        return np.array(A), np.array(b).reshape(-1,1)

    def compute_bayes_error(self, n_samples=5000):
        E = 0
        for k in range(self.K):
            A, b = self._form_constraints(k)
            Ek = self._integrate(A=A, b=b, n_samples=n_samples)
            E += self.marginals[k]*Ek
        self.bayes_error = 1-E
        return 1-E


class GaussianBayesDiag:
    def __init__(self, means, cov, marginals=None):
        self.means = means
        self.cov = cov
        self.cov_inv = 1. / self.cov
        self.K = len(means)
        if marginals is None:
            self.marginals = np.array([1./self.K]*self.K)

    def _integrate(self, A, b, n_samples=2000):
        lincon = lcg.LinearConstraints(A=A, b=b)
        subsetsim = lcg.multilevel_splitting.SubsetSimulation(linear_constraints=lincon,
                                                      n_samples=100,
                                                      domain_fraction=0.5,
                                                      n_skip=3)
        subsetsim.run(verbose=False)
        shifts = subsetsim.tracker.shift_sequence
        hdr = lcg.multilevel_splitting.HDR(linear_constraints=lincon,
                                   shift_sequence=shifts,
                                   n_samples=n_samples,
                                   n_skip=9,
                                   X_init=subsetsim.tracker.x_inits())
        hdr.run(verbose=False)
        hdr_integral = hdr.tracker.integral()
        return hdr_integral

    def _form_constraints(self, k):
        not_k_ix = [i for i in range(self.K) if i != k]
        mu_k = self.means[k]
        b = []
        A = []
        for j in not_k_ix:
            mu_j = self.means[j]
            bjk = np.dot(mu_j, np.multiply(self.cov_inv[:, None], mu_j)) - np.dot(mu_k, np.multiply(self.cov_inv[:, None], mu_k))
            Ajk = 2*(np.multiply(self.cov_inv[:, None], mu_k) - np.multiply(self.cov_inv[:, None], mu_j))
            bjk += np.dot(Ajk, mu_k)
            Ajk = np.multiply(np.sqrt(self.cov)[:, None], Ajk) # NOTE: this assumes covariance is diagonal
            b.append(bjk)
            A.append(Ajk)

        return np.array(A), np.array(b).reshape(-1,1)

    def compute_bayes_error(self, n_samples=5000):
        E = 0
        for k in range(self.K):
            A, b = self._form_constraints(k)
            Ek = self._integrate(A=A, b=b, n_samples=n_samples)
            E += self.marginals[k]*Ek
        self.bayes_error = 1-E
        return 1-E


class GaussianBayesLight:
    def __init__(self, means, cov, marginals=None):
        self.means = means
        self.cov = cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.K = len(means)
        if marginals is None:
            self.marginals = np.array([1. / self.K] * self.K)

    def _integrate(self, A, b, n_samples=2000):
        lincon = lcg.LinearConstraints(A=A, b=b)
        subsetsim = lcg.multilevel_splitting.SubsetSimulation(linear_constraints=lincon,
                                                              n_samples=100,
                                                              domain_fraction=0.5,
                                                              n_skip=3)
        subsetsim.run(verbose=False)
        shifts = subsetsim.tracker.shift_sequence
        hdr = lcg.multilevel_splitting.HDR(linear_constraints=lincon,
                                           shift_sequence=shifts,
                                           n_samples=n_samples,
                                           n_skip=9,
                                           X_init=subsetsim.tracker.x_inits())
        hdr.run(verbose=False)
        hdr_integral = hdr.tracker.integral()
        return hdr_integral

    def _form_constraints(self, k, fast=True, low_mem=False):
        not_k_ix = [i for i in range(self.K) if i != k]
        mu_k = self.means[k]
        if fast:
            # this is much more memory efficient since we never need to compute the full matrix A, but just the dot products comprising AA^T
            if low_mem:
                b = []
                AAT = np.empty(shape=(self.K - 1, self.K - 1))
                jj = 0
                for j in not_k_ix:
                    mu_j = self.means[j]
                    bjk = np.dot(mu_j, np.dot(self.cov_inv, mu_j)) - np.dot(mu_k, np.dot(self.cov_inv, mu_k))
                    Ajk = 2 * (np.dot(self.cov_inv, mu_k) - np.dot(self.cov_inv, mu_j))
                    bjk += np.dot(Ajk, mu_k)
                    Ajk = np.dot(np.sqrt(self.cov), Ajk)  # NOTE: this assumes covariance is diagonal
                    b.append(bjk)
                    ii = 0
                    for i in not_k_ix[:j]:
                        mu_i = self.means[i]
                        Aik = 2 * (np.dot(self.cov_inv, mu_k) - np.dot(self.cov_inv, mu_i))
                        Aik = np.dot(np.sqrt(self.cov), Aik).reshape(-1, 1)  # NOTE: this assumes covariance is diagonal
                        Aji = np.dot(Ajk, Aik)
                        AAT[jj, ii] = Aji
                        AAT[ii, jj] = Aji
                        ii += 1
                    jj += 1

                A = sqrtm(AAT)

                # compute the full matrix A, and then AA^T, uses lots of memory
            else:
                b = []
                A = []
                for j in not_k_ix:
                    mu_j = self.means[j]
                    bjk = np.dot(mu_j, np.dot(self.cov_inv, mu_j)) - np.dot(mu_k, np.dot(self.cov_inv, mu_k))
                    Ajk = 2 * (np.dot(self.cov_inv, mu_k) - np.dot(self.cov_inv, mu_j))
                    bjk += np.dot(Ajk, mu_k)
                    Ajk = np.dot(np.sqrt(self.cov), Ajk)  # NOTE: this assumes covariance is diagonal
                    b.append(bjk)
                    A.append(Ajk)
                A = np.array(A)
                A = sqrtm(A @ A.T)

            return A, np.array(b).reshape(-1, 1)

        # this is the original method, which is slow and memory inefficient
        else:
            b = []
            A = []
            for j in not_k_ix:
                mu_j = self.means[j]
                bjk = np.dot(mu_j, np.dot(self.cov_inv, mu_j)) - np.dot(mu_k, np.dot(self.cov_inv, mu_k))
                Ajk = 2 * (np.dot(self.cov_inv, mu_k) - np.dot(self.cov_inv, mu_j))
                bjk += np.dot(Ajk, mu_k)
                Ajk = np.dot(np.sqrt(self.cov), Ajk)  # NOTE: this assumes covariance is diagonal
                b.append(bjk)
                A.append(Ajk)

            return np.array(A), np.array(b).reshape(-1, 1)

    def compute_bayes_error(self, n_samples=5000, fast=True, low_mem=False):
        E = 0
        for k in range(self.K):
            A, b = self._form_constraints(k, fast=fast, low_mem=low_mem)
            # do it the fast way
            Ek = self._integrate(A, b=b, n_samples=n_samples)
            E += self.marginals[k] * Ek
        self.bayes_error = 1 - E
        return 1 - E


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mnist-ckpts/best.pth.tar",
        help="the model path you want to compute Bayes error for",
    )

    parser.add_argument("--temperature", type=float, default=1.0, help="temperature used to generate samples")

    parser.add_argument("--n_samples", type=int, default=60000, help="number of samples to use for LIN-ESS to estimate Bayes error integral")

    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))

    print('evaluating using model trained with ' + str(checkpoint['epoch']) + ' epochs')
    print('test loss is ' + str(checkpoint['test_loss']))

    net = checkpoint['net']

    num_class = net['module.means'].shape[0]
    means, cov = net['module.means'].view(num_class, -1), net['module.cov_diag'].abs().view(-1) * (args.temperature ** 2)

    gb = GaussianBayes(means=means.data.numpy(), cov=np.diag(cov.data.numpy()))
    err = gb.compute_bayes_error(n_samples=args.n_samples)

    print('Bayes Error is %f' % err)