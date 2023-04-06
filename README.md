# SCDAA Coursework 2022-23
-------------------------
**Name & Student Number:**\
Xinyu Zhao s2303292\
Jingzhi Kong s1882018\
Chenxi Hu s1823902

## Content

- [General Setup](#General-Setup)
- [Part 1: Linear quadratic regulator](#part-1-linear-quadratic-regulator)
    - [Exercise 1.1](#exercise-11)
    - [Exercise 1.2](#exercise-12)
- [Part 2: Supervised learning, checking the NNs are good enough](#part-2-supervised-learning-checking-the-nns-are-good-enough)
    - [Exercise 2.1](#exercise-21)
    - [Exercise 2.2](#exercise-22)
- [Part 3: Deep Galerkin approximation for a linear PDE](#part-3-deep-galerkin-approximation-for-a-linear-pde)
    - [Exercise 3.1](#exercise-31)
- [Part 4: Policy iteration with DGM](#part-4-policy-iteration-with-dgm)
    - [Exercise 4.1](#exercise-41)
- [References](#references)


## Part 1: Linear quadratic regulator<a name="part-1-linear-quadratic-regulator"></a>
We examine the following stochastic differential equation (SDE) for the state process $(X_s)_{s\in[t,T]}$:

\begin{equation}
dX_s = [HX_s + M\alpha_s] ds + \sigma dW_s, \quad s \in [t, T], \quad X_t = x. \tag{1}
\end{equation}

Our objective is to minimize the cost functional $J_\alpha(t, x)$ defined by

\begin{equation}
J_\alpha(t, x) := \mathbb{E}^{t,x}\left[\int^T_t (X^{\top}_s C X_s + \alpha^{\top}D\alpha_s) ds + X^{\top}_T R X_T\right],
\end{equation}

where $C \geq 0$, $R \geq 0$, and $D > 0$ are given deterministic $2 \times 2$ matrices. We seek the value function, denoted by $v(t, x)$:

\begin{equation}
v(t, x) := \inf_{\alpha} J_\alpha(t, x).
\end{equation}

By solving the associated Bellman partial differential equation (PDE), we obtain the expression for the value function:

\begin{equation}
v(t, x) = x^{\top}S(t)x + \int^T_t \operatorname{tr}(\sigma\sigma^{\top}S_r) dr,
\end{equation}

where $S$ is the solution to the Riccati ordinary differential equation (ODE):

\begin{equation}
\begin{aligned}
S'(r) &= -2H^{\top}S(r) + S_r MD^{-1}MS(r) - C, \quad r \in [t, T], \\
S(T) &= R.
\end{aligned}
\end{equation}

The solution $S$ takes values in the space of $2 \times 2$ matrices. Consequently, the optimal Markov control is given by

\begin{equation}
a(t, x) = -DM^{\top}S(t)x.
\end{equation}
### Exercise 1.1<a name="exercise-11"></a>

