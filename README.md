# Optimal Transport in NumPy

This repository contrains recent Optimal Transport Algorithms proposed in the following publications, implemented using NumPy.

[[1]](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf). Cuturi, M., 2013. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in neural information processing systems (pp. 2292-2300).

[[2]](https://arxiv.org/pdf/1802.04367.pdf). Dvurechensky, P., Gasnikov, A. and Kroshnin, A., 2018. Computational optimal transport: Complexity by accelerated gradient descent is better than by Sinkhorn's algorithm. arXiv preprint arXiv:1802.04367.

[[3]](https://papers.nips.cc/paper/9313-a-direct-tildeo1epsilon-iteration-parallel-algorithm-for-optimal-transport-supplemental.zip). Jambulapati, A., Sidford, A. and Tian, K., 1906. A direct o (1/ϵ) iteration parallel algorithm for optimal transport. ArXiv Preprint, 2019, p.2.

### Dependencies

- Python 3.6.4
- NumPy 1.14.2
- SciPy 1.1.0

### Quick start

- `sinkhorn_knopp.py` is the implementation of Sinkhorn’s Algorithm [1] presented in [this](https://github.com/rflamary/POT/blob/master/ot/bregman.py) repository. `sinkhorn_knopp_log.py` is the log-scale implementation of the same algorithm, which has better stability.

- `Approx_ot.py` and `AGD.py` correspond to Algorithm 2 and Algorithm 4 in [2], respectively.

- `ALG3.py` is Algorithm 3 as presented in the original paper [3]. `ALG3_v2.py` is a more stable version with optimized hyperparameters, which is based on a Julia implementation shared by the original authors.

- Run `test.py` to compute OT assignment between two MNIST digits using above algorithms, and compare their runtime and convergence.

### Acknowledgements

We use the base Sinkhorn’s Algorithm in [POT](https://github.com/rflamary/POT/blob/master/ot/bregman.py). We thank the authors of [3] for sharing their original implementation in Julia.
