# SCDAA Coursework 2022-23
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

## General SetUp<a name="General-Setup"></a>
```python
import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

```

## Part 1: Linear quadratic regulator<a name="part-1-linear-quadratic-regulator"></a>
We examine the following stochastic differential equation (SDE) for the state process $(X_s)_{s\in[t,T]}$:
$$dX_s = [HX_s + M\alpha_s] ds + \sigma dW_s, \quad s \in [t, T], \quad X_t = x.$$

Our objective is to minimize the cost functional $J^\alpha(t, x)$ defined by

$$J^\alpha(t, x) := \mathbb{E}^{t,x}\left[\int^T_t (X^{\top}_s C X_s + \alpha^{\top}D\alpha_s) ds + X^{\top}_T R X_T\right],$$

where $C \geq 0$, $R \geq 0$, and $D > 0$ are given deterministic $2 \times 2$ matrices. We seek the value function, denoted by $v(t, x)$:

$$v(t, x) := \inf_{\alpha} J^\alpha(t, x).$$

By solving the associated Bellman partial differential equation (PDE), we obtain the expression for the value function:

$$v(t, x) = x^{\top}S(t)x + \int^T_t \operatorname{tr}(\sigma\sigma^{\top}S_r) dr,$$

where $S$ is the solution to the Riccati ordinary differential equation (ODE):

$$
\begin{aligned}
S'(r) &= -2H^{\top}S(r) + S_r MD^{-1}MS(r) - C, \quad r \in [t, T], \\
S(T) &= R.
\end{aligned}
$$

The solution $S$ takes values in the space of $2 \times 2$ matrices. Consequently, the optimal Markov control is given by
$$a(t, x) = -DM^{\top}S(t)x.$$

### Exercise 1.1<a name="exercise-11"></a>
```python
    class LQR:
    """Linear Quadratic Regulator (LQR) for a linear time-invariant system with state-dependent noise."""

    def __init__(self, H, M, C, D, R, T, N, sigma):
        """Initializes the LQR problem with given system matrices, cost weights, time horizon, and noise standard deviation."""
        self.H = H  # state transition matrix
        self.M = M  # control matrix
        self.C = C  # state cost matrix
        self.D = D  # control cost matrix
        self.R = R  # final state cost matrix
        self.T = T  # time horizon
        self.sigma = sigma # state-dependent noise standard deviation
        self.N = N  # number of time step
       
        
    def solve_riccati(self, time_grid):
        """Solves the Riccati ode. """
        time_grid = torch.flip(time_grid, dims=[0])
        def riccati(S,t,H,M,D,C):
            S =  torch.tensor(S)
            S = torch.reshape(S,(2, 2))
            dSdt = -2 * H.T @ S + S @ M @ torch.inverse(D) @ M.T @ S - C
            return dSdt.flatten()
    
        # Terminal condition S(T)=R is the initial condition here
        S0=self.R
        
        # Slove ode
        S = odeint(riccati,S0.flatten(),time_grid,args=(self.H, self.M,self.D,self.C))
        
        S = torch.reshape(torch.tensor(S),(len(time_grid), 2, 2))
        S_reversed = torch.flip(S, dims=[0])

        return S_reversed.to(dtype=torch.float64)
        
    def value(self,time,space):
        """input: one torch tensor of dimension batch size (for time) and another torch tensor of dimension batch size × 1 × 2 (for space)
           return: a torch tensor of dimension batch size × 1 with entries being the control problem value v(t, x) for each t, x in the batch"""
        batch_size=time.size(0)
        end_time = torch.ones(batch_size) * self.T
        time_grid = torch.linspace(0, 1, self.N).unsqueeze(0).repeat(batch_size, 1)
        time_grid = time.view(-1, 1) + (end_time - time).view(-1, 1) * time_grid
        S_batch = torch.empty((batch_size, self.N, 2, 2),dtype=torch.float64)
        for i in range(batch_size):
            S_batch[i] = self.solve_riccati(time_grid[i])
        
        product = self.sigma @ torch.transpose(self.sigma, 0, 1)  @ S_batch
        temp = torch.diagonal(product, dim1=-2, dim2=-1).sum(dim=2)
        temp1 = space @ S_batch[:, 0, :, :] @ torch.transpose(space,1,2)
        v1=torch.reshape(temp1, (batch_size, 1))
        v2=torch.reshape(torch.trapz(temp, time_grid), (batch_size, 1))#[:, 0:1]
        v = v1+v2
        return v 
    
    def control(self, time, space):
        """input: one torch tensor of dimension batch size (for time) and another torch tensor of dimension batch size × 1 × 2 (for space)
           return: a torch tensor of dimension batch size × 2 with entries being the Markov control function for each t, x in the batch """
        batch_size = time.size(0)
        end_time = torch.ones(batch_size) * self.T
        time_grid = torch.linspace(0, 1, self.N).unsqueeze(0).repeat(batch_size, 1)
        time_grid = time.view(-1, 1) + (end_time - time).view(-1, 1) * time_grid
        S_batch = torch.empty((batch_size, self.N, 2, 2)).double()
        for i in range(batch_size):
            S_batch[i] = self.solve_riccati(time_grid[i])
        a = -torch.inverse(self.D.float()) @ self.M.T.float() @ S_batch[:, 0, :, :].float() @ torch.transpose(space, 1, 2).float()
        a = torch.reshape(a, (batch_size, 2))
        return a
    
    
    def control_sequence(self,time,space):
        """input: one torch tensor of dimension batch size (for time) and another torch tensor of dimension batch size × 1 × 2 (for space)
           return: a torch tensor of dimension batch size × 2 with entries being the Markov control function for each t, x in the batch """
        S_batch = torch.empty((time.size(0), time.size(1), 2, 2))
        for i in range(batch_size):
            S_batch[i] = self.solve_riccati(time[i])
        #a = ((-torch.inverse(D) @ M.T @  S_batch[:, 0, :, :]).repeat(num_samples*batch_size,1,1,)) @ torch.transpose(space,1,2)
        a = -torch.inverse(torch.tensor(self.D, dtype=torch.float64)) @ self.M.T.double() @ S_batch @ torch.tensor(space, dtype=torch.float64)
        a = torch.reshape(a,(-1, 2))
        return a
  ```
   
This code provides a Linear Quadratic Regulator (LQR) for a linear time-invariant system with state-dependent noise. It uses the Riccati equation to calculate the optimal control policy for a given set of system matrices and cost weights. The key features of the code include:

- Efficient numerical integration using the odeint function from PyTorch.
- Batch processing of multiple time-space inputs for increased speed and flexibility.
- Simple and intuitive API for calculating the optimal control value and control sequence.


**Usage**

To use the LQR control, you need to create an instance of the LQR class and provide it with the system matrices, cost weights, time horizon, and noise standard deviation. You can then use the value and control methods to calculate the optimal control value and control sequence for a given time-space input.


**Input format**

The value and control methods take two torch tensors as input:
- `time`: a tensor of dimension batch_size containing the time values for each input.
- `space`: a tensor of dimension batch_size × 1 × 2 containing the space values for each input.


**Output format**

The value and control methods return torch tensors of dimension batch_size × 1 and batch_size × 2, respectively.


**Classes and methods**

`init(self, H, M, C, D, R, T, N, sigma)`: Initializes the LQR problem with the given system matrices, cost weights, time horizon, and noise standard deviation. The input arguments are:

- `H` (numpy.ndarray): state transition matrix
- `M` (numpy.ndarray): control matrix
- `C` (numpy.ndarray): state cost matrix
- `D` (numpy.ndarray): control cost matrix
- `R` (numpy.ndarray): final state cost matrix
- `T` (float): time horizon
- `N` (int): number of time steps
- `sigma` (float or numpy.ndarray): state-dependent noise standard deviation.
- `solve_riccati(self, time_grid)`: Solves the Riccati ode using the initial condition S(T) = R. The input argument is:

- `time_grid (torch.Tensor)`: a 1D tensor of shape (N,) containing the time grid.
- `value(self, time, space)`: Computes the value function v(t, x) for the given inputs. The input arguments are:

- `time (torch.Tensor)`: a 1D tensor of shape (batch_size,) containing the time points.
- `space (torch.Tensor)`: a 3D tensor of shape (batch_size, 1, 2) containing the spatial coordinates.
- `control(self, time, space)`: Computes the Markov control function for the given inputs. 

The input arguments are:
- `time (torch.Tensor)`: a 2D tensor of shape (batch_size, N) containing the time grid.
- `space (torch.Tensor)`: a 3D tensor of shape (batch_size, N, 2) containing the spatial coordinates.

### Exercise 1.2<a name="exercise-12"></a>
```python
def mse(x, y):
    return torch.mean((x - y)**2)
# Vary number of time steps
num_samples =100000
num_steps = [1, 10, 50, 100, 500,1000,5000]#[100]

batch_size = 50
#t= torch.tensor([1,0.5,0.25], dtype=torch.float32)
#x =torch.tensor([[[1., 1.]],[[2., 3.]],[[1., 0.]]], dtype=torch.float32)
x =  torch.rand(batch_size,1, 2,dtype=torch.float64)
t=  torch.rand(batch_size,dtype=torch.float64)
T=1

errors = [] 
v_true = result.value(t,x)
v_true = torch.reshape(v_true,(1,-1))
for N in num_steps:
    end_time = torch.ones(batch_size, dtype=torch.float64) * T
    time_grid = torch.linspace(0, 1, N+1).unsqueeze(0).repeat(batch_size, 1)
    time_grid = t.view(-1, 1) + (end_time - t).view(-1, 1) * time_grid
    S_batch = torch.empty((batch_size, N+1, 2, 2),dtype=torch.float64)
    for i in range(batch_size):
        S_batch[i] = result.solve_riccati(time_grid[i])
    X =torch.reshape(x.repeat(num_samples, 1, 1),(num_samples,batch_size ,2, 1))#num_samples,batch_size,num_steps+1,2,1
    tau=(T-t)/N
    v_temp = 0
    X_new = 0
    for n in range(0, N):
        a  = -torch.inverse(D) @ M.T @ S_batch[:,n,:,:] @ X
        X_new =  X + (H @ X + M @ a )*(tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) + torch.reshape(torch.transpose(sigma*torch.reshape(torch.randn(num_samples,batch_size) *  torch.sqrt(tau),(1,num_samples*batch_size)),0,1),(num_samples,batch_size,2,1))
        v_temp +=( torch.transpose(X,2,3)@ (C @X)+torch.transpose(a,2,3) @ (D @ a)) *(tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        X = X_new
    v_temp += torch.transpose(X,2,3)@ (R @X)
    MC_results = torch.mean(v_temp, dim=0)
    errors.append(mse(torch.reshape(MC_results,(1,batch_size)), v_true))
log_num_samples_list = np.log(num_steps)
log_errors = np.log(errors)
plt.plot(log_num_samples_list, log_errors, marker='o',linestyle='-')
plt.xlim([-1, 10])
plt.xlabel('Number of Steps', fontsize=14)
plt.ylabel('Error', fontsize=14)
```
This code is related to Exercise 1.2 of the Optimal Control course. This code calculates the mean squared error of a Monte Carlo simulation used to solve a financial problem. The problem involves solving a Riccati differential equation, and the code simulates a number of paths of the underlying asset price to obtain an estimate of the value function. The error of the simulation is calculated for different numbers of time steps.

**Usage**

To use this code, simply run the mse function with the appropriate input values. The function takes two PyTorch tensors, x and y, and returns the mean squared error between them.

The main part of the code involves varying the number of time steps and calculating the error of the Monte Carlo simulation for each number of steps. The simulation involves solving a Riccati differential equation and simulating a number of paths of the underlying asset price. The error of the simulation is then calculated and plotted against the number of time steps.

The code follows the following steps:

- Define a mean-square error function, `mse(x, y)`, that calculates the mean of the squared difference between two tensors `x` and `y`.
- Define the number of samples and number of time steps to use in the simulation. The number of time steps varies from a single step to 5000 steps.
- Define the size of the batch of random inputs to use in the simulation. Each input in the batch is a vector of length 2.
- Define the time grid for the simulation as a tensor of size `batch_size x (N+1)`, where `N` is the number of time steps. Each row of the tensor corresponds to a different input in the batch and represents the time grid for that input.
- For each input in the batch, solve the Riccati differential equation using the `solve_riccati` method of an LQRResult object called result. The result is stored in a tensor of size `batch_size x (N+1) x 2 x 2`.
- Initialize the state variable X as a tensor of size `num_samples x batch_size x 2 x 1`. This represents the current state of the system for each sample and input in the batch.
- Define the time step size `tau` as the difference between the final time `T` and the initial time t divided by the number of time steps `N`.
- For each time step n, calculate the control input a using the current state X and the Riccati solution `S_batch[:,n,:,:]`. Then update the state variable `X` using the control input and a Gaussian noise term.
- Calculate the final cost using the updated state variable `X` and the terminal cost matrix `R`.
- Calculate the Monte Carlo estimate of the optimal value function by averaging the final costs over all samples.
- Calculate the mean-squared error between the Monte Carlo estimate and the true value function for the given inputs.
- Plot the error as a function of the number of time steps used in the simulation.

**Result**

The code generates a plot that shows the error as a function of the number of time steps used in the simulation. The plot shows that the error decreases as the number of time steps increases, which is expected.

```python
def mse(x, y):
    return torch.mean((x - y)**2)
# Vary number of time steps
num_samples_list =[10, 50,100,5*10**2,10**3, 5*10**3, 10**4, 5*10**4, 10**5]
N =5000
batch_size = 50
x =  torch.rand(batch_size,1, 2,dtype=torch.float64)
t=  torch.rand(batch_size,dtype=torch.float64)
errors = [] 
v_true = result.value(t,x)
v_true = torch.reshape(v_true,(1,-1))
for num_samples in num_samples_list:
    end_time = torch.ones(batch_size, dtype=torch.float64) * T
    time_grid = torch.linspace(0, 1, N+1).unsqueeze(0).repeat(batch_size, 1)
    time_grid = t.view(-1, 1) + (end_time - t).view(-1, 1) * time_grid
    S_batch = torch.empty((batch_size, N+1, 2, 2),dtype=torch.float64)
    for i in range(batch_size):
        S_batch[i] = result.solve_riccati(time_grid[i])
    X =torch.reshape(x.repeat(num_samples, 1, 1),(num_samples,batch_size ,2, 1))#num_samples,batch_size,num_steps+1,2,1
    tau=(T-t)/N
    v_temp = 0
    X_new = 0
    for n in range(0, N):
        a  = -torch.inverse(D) @ M.T @ S_batch[:,n,:,:] @ X
        X_new =  X + (H @ X + M @ a )*(tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))+ torch.reshape(torch.transpose(sigma*torch.reshape(torch.randn(num_samples,batch_size)*  torch.sqrt(tau),(1,num_samples*batch_size)),0,1),(num_samples,batch_size,2,1))
        v_temp += ( torch.transpose(X,2,3)@ (C @X)+torch.transpose(a,2,3) @ (D @ a)) *(tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        X = X_new
    v_temp += torch.transpose(X,2,3)@ (R @X)
    MC_results = torch.mean(v_temp, dim=0)
    errors.append(mse(torch.reshape(MC_results,(1,batch_size)), v_true))
log_num_samples_list = np.log(num_samples_list)
log_errors = np.log(errors)
plt.plot(log_num_samples_list, log_errors, marker='o',linestyle='-')
plt.xlim([2, 12])
plt.xlabel('Number of Monte Carlo samples', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.show()
```

**Usage**

To use this code, simply run the Python script. The simulation parameters can be adjusted by changing the values of `num_samples_list`, `N`, `batch_size`, `x`, and `t`. The resulting plot shows the mean squared error between the true value of the solution and the estimated value of the solution for each value of `num_samples`.

The code defines a function `mse(x, y)` that calculates the mean squared error between two tensors `x` and y. The simulation is run for each value of `num_samples` in `num_samples_list`. The simulation uses the `solve_riccati` function from the result module to solve the Riccati equation, and the resulting solution is used to simulate the control problem using Monte Carlo methods. The estimated value of the objective function is calculated using the mean of the temporary value of the objective function over all Monte Carlo samples. The mean squared error between the estimated value and the true value is then calculated and plotted against the number of Monte Carlo samples using a log-log plot.

**Result**

The resulting plot shows how the error decreases as the number of Monte Carlo samples increases. It can be used to determine the number of Monte Carlo samples needed to achieve a desired level of accuracy in the simulation.


## Part 2: Supervised learning, checking the NNs are good enough<a name="part-2-supervised-learning-checking-the-nns-are-good-enough"></a>
In part 2, a neural network is defined as a parametric function that depends on an input, denoted as $x \in \mathbb{R}^d$, and parameters, denoted as $\theta \in \mathbb{R}^p$, taking values in $\mathbb{R}^{d'}$. We write this function as:

$$\phi = \phi(x; \theta).$$

An example of a one-hidden-layer neural network is given by

$$\phi(x; \theta) = \phi(x; \alpha^{(1)}, \alpha^{(2)}, \beta^{(1)}, \beta^{(2)}) = \alpha^{(1)}\psi(\alpha^{(2)}x + \beta^{(2)}) + \beta^{(1)},$$

where $\psi$ is an activation function applied component-wise, $\alpha^{(2)}$ is an $h \times d$ matrix, $\beta^{(2)}$ is an $h$-dimensional vector, $\alpha^{(1)}$ is a $d' \times h$ matrix, and $\beta^{(1)}$ is a $d'$-dimensional vector. In this case, $\theta = (\alpha^{(1)}, \alpha^{(2)}, \beta^{(1)}, \beta^{(2)})$, and $p = h \times d + h + d' \times h + d'$ represents the number of parameters or "weights" depending on the size of the hidden layer $h$.

The most relevant supervised learning task for our purposes is as follows: We aim to find neural network (NN) weights $\theta^*$ such that our NN $\phi(\cdot; \theta^*)$ is a good approximation of some function $f$. We are given a training dataset $\{(x^{(i)}, f(x^{(i)}))\}^{N_{\text{data}}}_{i=1}$ and search for $\theta^*$ by attempting to minimize

$$R(\theta) := \frac{1}{N_{\text{data}}} \sum^{N_{\text{data}}}_{i=1} \left|\phi(x^{(i)}; \theta) - f(x^{(i)})\right|^2$$

over $\theta \in \mathbb{R}^p$ by running some variant of a gradient descent algorithm. For example, starting with an initial guess of $\theta^{(0)}$, we update $\theta^{(k+1)}$ as


$$\theta^{(k+1)} = \theta^{(k)} - \gamma \nabla_\theta R(\theta^{(k)}), \quad k = 0, 1, 2, \dots.$$

### Exercise 2.1<a name="exercise-21"></a>
```python
class DGM_Layer(nn.Module):
    
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
            

        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)
            
    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut), self.activation)
        return l
    
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        
        output = ((1-G))*H + Z*S
        return output


class Net_DGM(nn.Module):

    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        output = self.output_layer(S4)
        return output
 ```
This repository contains the implementation of a Deep Galerkin Method (DGM) for solving partial differential equations (PDEs) using PyTorch. The code consists of two classes: `DGM_Layer` and `Net_DGM`.

`DGM_Layer` is a class that represents a single DGM layer. It takes as input the dimensions of the input data (`dim_x`) and the latent space (`dim_S`), as well as the activation function to be used (activation). It implements the DGM equations using four gate functions: `gate_Z`, `gate_G`, `gate_R`, and `gate_H`. The output of the layer is computed as ((1-G))H + ZS.

`Net_DGM` is a class that represents the entire DGM network. It takes as input the dimensions of the input data (`dim_x`) and the latent space (`dim_S`), as well as the activation function to be used (activation). It consists of an input layer, three `DGM_Layer` layers (DGM1, DGM2, and DGM3), and an output layer. The input layer is a single fully connected layer followed by the activation function. The output layer is a single linear layer that outputs a scalar value. The DGM layers are used to learn the latent representation of the input data.

**Usage**

To use this code, first create an instance of the Net_DGM class by specifying the input and output dimensions of the DGM, as well as the activation function to be used in the neural networks. Then, train the DGM on a set of input/output pairs using a suitable optimization algorithm, such as stochastic gradient descent (SGD).

Once the DGM is trained, it can be used to predict the solution of the PDE at a given point in time and space by calling the forward method of the `Net_DGM` instance, passing in the time and space coordinates as inputs.

```python
# Generate training data
N_data = 2000
t_train = (torch.rand(N_data) * T).double()
x_train = (torch.rand(N_data, 2) * 6 - 3).double()
v_train = (result.value(t_train, x_train.unsqueeze(1))).double()

# Create neural network
dim_x = 2
dim_S = 100
net = Net_DGM(dim_x, dim_S).double()

# Set up the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(t_train.unsqueeze(1), x_train)
    loss = loss_fn(outputs, v_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

# Plot the training loss
plt.figure(figsize=(15, 8))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

This code generates training data and trains a neural network using the Deep Galerkin Method to solve a partial differential equation.

The code generates random training data with N_data data points and applies the DGM neural network to fit a function defined by the result.value function. The neural network is defined by the `Net_DGM` class and uses three `DGM_Layer` classes to perform the approximation. The optimizer is set up using the `optim.Adam` function with a learning rate of 0.0001, and a MultiStepLR scheduler is used to decay the learning rate. The loss function used is the mean squared error loss, defined by `nn.MSELoss`.

The training loop is defined by num_epochs and the loss function is minimized using backpropagation. The loss at each epoch is recorded and the training loss is plotted using matplotlib.

Overall, this code provides an example of how to use the DGM method to solve a partial differential equation using PyTorch.

### Exercise 2.2<a name="exercise-22"></a>
```python
class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()
        
        layers = [nn.BatchNorm1d(sizes[0]),] if batch_norm else []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]).double())
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j+1], affine=True).double())
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True

    def forward(self, x):
        return self.net(x)
```
This code defines a PyTorch nn.Module called `FFN` that represents a feed-forward neural network. The module constructor takes in an array sizes that specifies the number of neurons in each layer of the network. Additionally, the module can be initialized with an activation function and an `output_activation` function, both defaulting to nn.ReLU and nn.Identity, respectively. Finally, there is an optional flag `batch_norm` that when set to True adds batch normalization layers between the input and each hidden layer.

The forward method of the module takes in a tensor `x` as input and applies the neural network to it. The freeze and unfreeze methods set all or no parameters in the network to require gradients during training, respectively.

This module is provided to facilitate the construction of feed-forward neural networks in PyTorch.

```python
lqr = LQR(H, M, C, D, R, T, 100, sigma)

# Problem parameters
T = 1
num_epochs = 1000

# Generate training data
N_data = 2000
t_data = np.random.uniform(0, T, size=(N_data, 1)).astype(np.float64)
x_data = np.random.uniform(-3, 3, size=(N_data, 2)).astype(np.float64)
a_data = np.array([lqr.control(torch.tensor(t, dtype=torch.float64), torch.tensor(x, dtype=torch.float64).view(1, 1, 2)).detach().numpy() for t, x in zip(t_data, x_data)])

# Create neural network
sizes = [3, 100, 100, 2]  # input_dim=3 (t, x1, x2), output_dim=2
net = FFN(sizes, activation=nn.ReLU, output_activation=nn.Identity)

# Set up the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Define the MSE loss
mse_loss = nn.MSELoss()

# Training loop
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Use the entire dataset
    t_batch = torch.tensor(t_data, dtype=torch.float64)
    x_batch = torch.tensor(x_data, dtype=torch.float64)
    a_batch = torch.tensor(a_data, dtype=torch.float64)

    # Forward pass
    tx_batch = torch.cat((t_batch, x_batch), dim=1)
    a_pred = net(tx_batch)
    #Add dimension to a_pred
    a_pred = a_pred.unsqueeze(dim=1)

    # Calculate loss using MSE
    loss = mse_loss(a_pred, a_batch)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())


# Plot the training loss
plt.figure(figsize=(15, 8))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```
This code provides an implementation of a linear-quadratic regulator (LQR) controller using neural networks. The LQR controller is designed to control a linear time-invariant system with state space representation, given by the matrices `H`, `M`, `C`, and `D`. The neural network is trained using a dataset of inputs and outputs generated by the LQR controller.

**Usage**

1. Create an instance of the LQR controller by providing the matrices `H`, `M`, `C`, `D`, `R`, `T`, and `sigma`.
2. Generate training data by randomly sampling `N_data` time and state pairs, and computing the LQR control action for each pair using the lqr.control function.
3. Create a neural network using the `FFN` class, with the desired architecture and activation functions.
4. Set up the optimizer and loss function.
5. Train the neural network using the generated training data, optimizing for the MSE loss.
6. Plot the training loss.

## Part 3: Deep Galerkin approximation for a linear PDE<a name="part-3-deep-galerkin-approximation-for-a-linear-pde"></a>
Consider the linear partial differential equation (PDE) with $\alpha = (1, 1)^\top$ and $H, M, C, D, R, \sigma$ as the matrices from Exercise (1.1):

$$
\begin{aligned}
&\partial_t u + \frac{1}{2} \operatorname{tr}(\sigma\sigma^{\top}\partial_{xx}u) + Hx\partial_x u + M\alpha\partial_x u + x^{\top}Cx + \alpha^{\top}D\alpha = 0 &\text{ on } [0, T) \times \mathbb{R}^2, \\
&u(T, x) = x^{\top}Rx &\text{ on } \mathbb{R}^2.
\end{aligned}
$$

This PDE represents the linearization of the Bellman PDE resulting from taking the constant control $\alpha = (1, 1)^\top$, regardless of the state of the system. The deep Galerkin method replaces $u$ with a neural network approximation $u(\cdot, \cdot; \theta)$, selects random points from the problem domain $(t^{(i)}, x^{(i)})$ with $i = 1, \dots, N_{\text{batch}}$, and aims to minimize

$$R(\theta) := R_{\text{eqn}}(\theta) + R_{\text{boundary}}(\theta),$$

where

$$
\begin{aligned}
R_{\text{eqn}}(\theta) &= \frac{1}{N_{\text{batch}}} \sum^{N_{\text{batch}}}_{i=1} \left|\partial_t u(t^{(i)}, x^{(i)}; \theta) + \frac{1}{2}\operatorname{tr}(\sigma\sigma^{\top}\partial_{xx}u(t^{(i)}, x^{(i)}; \theta)) + Hx^{(i)}\partial_x u(t^{(i)}, x^{(i)}; \theta) \right. \\
&\qquad \left. + M\alpha\partial_x u(t^{(i)}, x^{(i)}; \theta) + (x^{(i)})^{\top}Cx^{(i)} + \alpha^{\top}D\alpha\right|^2, \\
R_{\text{boundary}}(\theta) &= \frac{1}{N_{\text{batch}}} \sum^{N_{\text{batch}}}_{i=1} \left|u(T, x^{(i)}; \theta) - (x^{(i)})^{\top}Rx^{(i)}\right|^2,
\end{aligned}
$$

over $\theta \in \mathbb{R}^p$. Since the right-hand side of the PDE we are solving is zero, if we can achieve $R(\theta) = 0$, we have a good approximation of the solution.

### Exercise 3.1<a name="exercise-31"></a>
```python
def MC_results(x, t, num_samples, num_steps, num_loops, T):
    num_samples_per_batch = int(num_samples /num_loops)
    batch_size = t.size(0)
    for N in num_steps:
        end_time = torch.ones(batch_size, dtype=torch.float64) * T
        time_grid = torch.linspace(0, 1, N+1).unsqueeze(0).repeat(batch_size, 1)
        time_grid = t.view(-1, 1) + (end_time - t).view(-1, 1) * time_grid
        S_batch = torch.empty((batch_size, N+1, 2, 2),dtype=torch.float64)
        a = torch.empty((num_samples_per_batch,batch_size,N+1,2,1),dtype=torch.float64)
        a[:,:,:,:,:]=torch.tensor([[1],[1]],dtype=torch.float64)
        for i in range(batch_size):
            S_batch[i] = result.solve_riccati(time_grid[i])
        for batch_index in range(num_loops):
            X = torch.empty(num_samples_per_batch,batch_size , N+1, 2,1,dtype=torch.float64)#num_samples,batch_size,num_steps+1,2,1
            X[:,:, 0, :, :] = torch.reshape(x.repeat(num_samples_per_batch, 1, 1).unsqueeze(0),(num_samples_per_batch,batch_size,2,1))
            tau=(T-t)/N
            for n in range(0, N):
                X[:, :,n+1, :, :] =  X[:, :,n, :, :] + (H @ X[:, :,n, :, :] + M @  a[:, :,n, :, :] )*(tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))+ torch.reshape(torch.transpose(sigma*torch.reshape(torch.randn(num_samples_per_batch,batch_size)*  torch.sqrt(tau),(1,num_samples_per_batch*batch_size)),0,1),(num_samples_per_batch,batch_size,2,1))
            j_temp=torch.transpose(X,3,4) @ (C @X)+torch.transpose(a,3,4) @ (D @ a)
            j=torch.trapz(torch.reshape(j_temp,(num_samples_per_batch,batch_size,N+1)),torch.reshape(time_grid.repeat(num_samples_per_batch,1),(num_samples_per_batch,batch_size,N+1)))+ torch.reshape(torch.transpose(X[:,:,N,:,:],2,3) @ (R @ X[:,:,N,:,:]),(num_samples_per_batch,batch_size))
            MC_results_batch = torch.mean(j, dim=0)
        # Concatenate the results for this batch to the overall results
            if batch_index == 0:
                MC_results = MC_results_batch
            else:
                MC_results = torch.cat((MC_results, MC_results_batch))
        v_MC = torch.mean(torch.reshape(MC_results,(num_loops,batch_size)),dim=0)
    return v_MC
 ```
 
This Python code calculates Monte Carlo results for a given set of parameters. The `MC_results` function takes in the following arguments:
- `x`: the initial state of the system
- `t`: a tensor containing the start time of the simulation
- `num_samples`: the total number of samples to generate
- `num_steps`: a list of the number of steps to take in the simulation
- `num_loops`: the number of times to repeat the simulation for each batch
- `T`: the end time of the simulation
The function then calculates the Monte Carlo results by iterating through the provided number of steps, generating batches of samples, and concatenating the results. Finally, the function returns the mean of the overall results.

To use the code, simply call the `MC_results` function with the desired arguments. Make sure to import the necessary libraries and ensure that the inputs are of the correct data types.

Note that this code requires the PyTorch library to be installed.
```python
def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_laplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:, d].view(-1, 1)
        grad2 = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2[:, d].view(-1, 1))
    hess_diag = torch.cat(hess_diag, 1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian
def get_hess(grad, x):
    hess = torch.empty((x.shape[0],2,2))
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess[:,d,:] = grad2
    
    return hess
```
```python
    # Neural network
dim_x = 2
dim_S = 100
net_3 = Net_DGM(dim_x, dim_S).double()
T = 1
batch_size = 200
t_3_batch  = (torch.rand(batch_size, requires_grad=True) * T).double()
x_3_batch= (torch.rand(batch_size,2, requires_grad=True) * 2 - 1).double()
MC = MC_results(x_3_batch.detach(), t_3_batch.detach(), 100, [100], 1, T)
optimizer = torch.optim.Adam(net_3.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)

loss_fn = nn.MSELoss()

alpha = torch.tensor([[1.0], [1.0]], dtype=torch.float64)#.clone().detach()
max_updates = 500
losses = []
error_history = []
for it in range(max_updates):
    optimizer.zero_grad()
    
    u_of_tx = net_3(t_3_batch.unsqueeze(1), x_3_batch)
    grad_u_x = get_gradient(u_of_tx,x_3_batch)
    grad_u_t = get_gradient(u_of_tx,t_3_batch)
    grad_u_xx = get_hess(grad_u_x, x_3_batch)
    
    trace = torch.matmul(sigma.view(1, 2, 1).double(), sigma.view(1, 1, 2).double()) @ grad_u_xx.double()
    target_functional = torch.zeros_like(u_of_tx)

    pde = grad_u_t\
      + 0.5 * ((trace[:,0,0]) + (trace[:,1,1])) \
      +  torch.matmul(torch.matmul(grad_u_x.unsqueeze(1), H), x_3_batch.unsqueeze(2))[:,0,0] \
      + ((grad_u_x).unsqueeze(1) @ M @ alpha)[:,0,0] \
      + ((x_3_batch).unsqueeze(1) @ (C @ x_3_batch.unsqueeze(2)))[:,0,0] \
      + (alpha.T @ D @ alpha)


    MSE_functional = loss_fn(torch.reshape(pde,(batch_size,1)), target_functional)
    

    # Compute the terminal condition residual       
    
    x_batch_terminal = x_3_batch#(torch.rand(batch_size, 2, requires_grad=True) * 6 - 3).double()
    t_batch_terminal =(torch.ones(batch_size) * T).double()
    u_of_Tx = net_3(t_batch_terminal.unsqueeze(1), x_batch_terminal)
    target_terminal = x_batch_terminal.unsqueeze(1) @ R @ x_batch_terminal.unsqueeze(2)
    MSE_terminal = loss_fn(u_of_Tx, target_terminal[:,:,0])
    
    # Compute total loss
    loss = MSE_functional + MSE_terminal
    loss.backward(retain_graph=True)
    losses.append(loss.item())

    # Backpropagation and optimization
    
    optimizer.step()
    scheduler.step()
    # Compute error against Monte Carlo solution
    if it % 100 == 0:
        # Replace your Monte Carlo computation here, using alpha instead of optimal control
        # mc_results = ...
        error = loss_fn(MC.unsqueeze(1),u_of_tx)
        error_history.append(error.item())
plt.figure(figsize=(15, 8))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(error_history)
plt.xlabel('Iteration // 100')
plt.ylabel('error')
plt.title('Monte Carlo')
plt.show()
    return hess
```

This exercise involves implementing the Deep Galerkin Method (DGM) for the given linear partial differential equation (PDE) and comparing the neural network-based approximation of the solution against the Monte Carlo solution at regular intervals during the training. The goal is to approximate the solution of the PDE using a neural network and the Deep Galerkin Method.

**Usage**

To run the provided code for Exercise 3.1, follow these steps:
- Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

- Copy the provided code to a Python script or Jupyter Notebook.

- Run the script or notebook, and observe the training loss and Monte Carlo error plots.

- (Optional) Modify the neural network architecture, optimizer settings, or other parameters to improve the performance or experiment with different configurations.

The provided code sets up a neural network architecture and performs the following steps:

- Initialize the neural network with the given dimensions and create an optimizer and learning rate scheduler.

- Generate random points from the problem domain as training data.

- Calculate the Monte Carlo solution for the given problem by adapting the method from Exercise 1.2.

- Perform the training process for a specified number of iterations (`max_updates`). In each iteration:

    - Calculate the gradients of the neural network output with respect to the input variables (`x` and `t`) and the second-order gradients (Hessian) of the output with respect to `x`.
 
    - Compute the PDE residual using the given formula and calculate the mean squared error (MSE) between the computed residual and the target functional (zero in this case).
    - Compute the terminal condition residual and its MSE with respect to the target terminal condition.

    - Add the two MSEs to get the total loss, perform backpropagation, and update the neural network weights.

- Calculate the error between the neural network-based solution and the Monte Carlo solution at regular intervals during the training process and store the errors in the `error_history` list.

- Plot the training loss and the Monte Carlo error as a function of the iteration number.

**Results**

The provided code will output two plots:

- The training loss plot, which shows the decrease in the loss function as the neural network is trained over multiple iterations.

- The Monte Carlo error plot, which shows the error between the neural network-based approximation and the Monte Carlo solution at regular intervals during the training process.

## References<a name="references"></a>
- M. Sabate-Vidales, Deep-PDE-Solvers, Github project,https://github.com/msabvid/Deep-PDE-Solvers, 2021.
- C. Jiang, Deep Galerkin Method, Github project, https://github.com/EurasianEagleOwl/DeepGalerkinMethod, 2022.
- G. dos Reis and D. Sˇiˇska. Stochastic Control and Dynamic Asset Allocation.https://www.maths.ed.ac.uk/∼dsiska/LecNotesSCDAA.pdf. 2021.
