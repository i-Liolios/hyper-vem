# Mathematical Derivations

This document outlines the mathematical formulation of the Variational Expectation-Maximization (VEM) algorithm used in `hypervem`.

## Notation

* **Number of nodes:** $N$
* **Number of hyperedges:** $M$
* **Number of node clusters:** $K$
* **Number of hyperedge clusters:** $G$
* **Prior (proportion) of node clusters:** $\gamma = (\gamma_1, \dots, \gamma_K)$
* **Prior (proportion) of hyperedge clusters:** $\delta = (\delta_1, \dots, \delta_G)$
* **Connection probability matrix:** $\Theta = (\theta_{kg}) \in [0,1]^{K \times G}$
* **Incidence matrix (observed data):** $X \in \{0,1\}^{M \times N}$
* **Soft node cluster assignments:** $c_i(k) = P(\text{Cluster of node i} = k)$
* **Soft hyperedge cluster assignments:** $d_j(g) = P(\text{Cluster of hyperedge j} = g)$

---

## ELBO

The objective function maximized by the VEM algorithm is the Evidence Lower Bound (ELBO), which is a function of the variational parameters ($c, d$) and model parameters ($\gamma, \delta, \Theta$):

$$
\begin{align}
ELBO(\boldsymbol{c}, \boldsymbol{d}, \boldsymbol\gamma, \boldsymbol\delta, \Theta) &= \sum_{j=1}^M\sum_{g=1}^G d_j(g)\log(\delta_g) + \sum_{i = 1}^N\sum_{k=1}^K c_i(k)\log(\gamma_k) \\
&+ \sum_{j=1}^M\sum_{g=1}^G\sum_{i = 1}^N\sum_{k=1}^K d_j(g)c_i(k) \log\left(\theta_{kg}^{x_{ji}}(1-\theta_{kg})^{1-x_{ji}}\right) \\
&- \sum_{i=1}^N\sum_{k=1}^K c_i(k)\log(c_i(k)) - \sum_{j=1}^M\sum_{g=1}^G d_j(g)\log(d_j(g))
\end{align}
$$

---

## VE-step

The optimization of the ELBO with respect to the variational parameters ($c, d$) yields the following closed-form updates:

$$
\hat{c}_i(k) = \frac{\gamma_k \prod\limits_{g=1}^G \theta_{kg}^{q_{ig}}(1 - \theta_{kg})^{b_g - q_{ig} }}{\sum\limits_{k'=1}^K\gamma_{k'} \prod\limits_{g=1}^G \theta_{k'g}^{q_{ig}}(1 - \theta_{k'g})^{b_g - q_{ig}}}
$$

$$
\hat{d}_j(g) = \frac{\delta_g \prod\limits_{k=1}^K \theta_{kg}^{y_{jk}}(1 - \theta_{kg})^{s_k - y_{jk} }}{\sum\limits_{g'=1}^G\delta_{g'} \prod\limits_{k=1}^K \theta_{kg'}^{y_{jk}}(1 - \theta_{kg'})^{s_k - y_{jk} }}
$$

**Auxiliary variables:**

$$
\begin{align}
q_{ig} &= \sum_{j=1}^M d_j(g)x_{ji} & y_{jk} &= \sum_{i=1}^N c_i(k)x_{ji} \\
b_g &= \sum_{j=1}^M d_j(g) & s_k &= \sum_{i=1}^N c_i(k)
\end{align}
$$

---

## M-step

The maximization of the ELBO with respect to the model parameters ($\gamma, \delta, \Theta$) yields closed-form updates:

$$
\hat\gamma_k = \frac{\sum_{i=1}^N c_i(k)}{N}
$$

$$
\hat\delta_g = \frac{\sum_{j=1}^M d_j(g)}{M}
$$

$$
\hat\theta_{kg} = \frac{\sum_{j=1}^M \sum_{i=1}^N x_{ji} d_j(g) c_i(k)}{\sum_{j=1}^M \sum_{i=1}^N d_j(g) c_i(k)}
$$
