# SCDAA Coursework 2022-23
**Name & Student Number:**\
Xinyu Zhao s2303292\
Jingzhi Kong s1882018\
Chenxi Hu s1823902

This is an instruction. We explain in detail how to run the codes for every exercise and how to modify its parameters, but some of them may be complicated. If you encounter running problems, you can try run all. Although it may take some time, there will be no errors and warning, and all the results will be displayed correctly. :)

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

```python
H = torch.tensor([[1,0],[0,1]], dtype=torch.float64)
M = torch.tensor([[1,0],[0,1]], dtype=torch.float64)
sigma = torch.tensor([[0.05],[0.05]], dtype=torch.float64)
C = torch.tensor([[0.1,0],[0,0.1]], dtype=torch.float64)
D = torch.tensor([[0.1,0],[0,0.1]], dtype=torch.float64)
R = torch.tensor([[1,0],[0,1]], dtype=torch.float64)
T = 1
#time_grid=torch.linspace(0, T, 6)
if __name__ == '__main__': 
    result = LQR(H, M, C, D, R, T, 100,sigma) 
    time = torch.rand(5, dtype=torch.float64)*T
    space = torch.rand(5, 1, 2, dtype=torch.float64)
    result.value(time,space)
    result.control(time,space)
``` 
   
This code provides a Linear Quadratic Regulator (LQR) for a linear time-invariant system with state-dependent noise. It uses the Riccati equation to calculate the optimal control policy for a given set of system matrices and cost weights. The key features of the code include:

- Efficient numerical integration using the odeint function from PyTorch.
- Batch processing of multiple time-space inputs for increased speed and flexibility.
- Simple and intuitive API for calculating the optimal control value and control sequence.


**Usage**

Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

To use this code, you need to create an instance of the LQR class and provide it with the paramters H, M, C, D, R, T, N, sigma, where H is symmetrical. You can then use the value and control methods to calculate the optimal control value and control sequence for a given time-space input.

The second code is an example of using. The following parts use the same parameters H, M, C, D, R, T, N, sigma.

**Input format**

The value and control methods take two torch tensors as input:
- `time`: a tensor of dimension batch_size containing the time values for each input.
- `space`: a tensor of dimension batch_size × 1 × 2 containing the space values for each input.


**Output format**

The value and control methods return torch tensors of dimension batch_size × 1 and batch_size × 2, respectively.


**Classes and methods**

`init(self, H, M, C, D, R, T, N, sigma)`: Initializes the LQR problem with the given parameters. The input arguments are:

- `H` (numpy.ndarray): state transition matrix
- `M` (numpy.ndarray): control matrix
- `C` (numpy.ndarray): state cost matrix
- `D` (numpy.ndarray): control cost matrix
- `R` (numpy.ndarray): final state cost matrix
- `T` (float): time horizon
- `N` (int): number of time steps
- `sigma` (float or numpy.ndarray): state-dependent noise standard deviation.

- `solve_riccati(self, time_grid)`: Solves the Riccati ode using the terminal condition S(T) = R. The input argument is:
- `time_grid (torch.Tensor)`: a 1D tensor of shape (N,) containing the time grid from t to T, time steps is N.
- The output is a 3D tensor of shape (N, 2, 2) containing the solutions of Riccati solution with respect to every time step.

- `value(self, time, space)`: Computes the value function v(t, x) for the given inputs. The input arguments are:
- `time (torch.Tensor)`: a 1D tensor of shape (batch_size,) containing the time points.
- `space (torch.Tensor)`: a 3D tensor of shape (batch_size, 1, 2) containing the spatial coordinates.

- `control(self, time, space)`: Computes the Markov control function for the given inputs. The input arguments are:
- `time (torch.Tensor)`: a 1D tensor of shape (batch_size, ) containing the time grid.
- `space (torch.Tensor)`: a 3D tensor of shape (batch_size, 1, 2) containing the spatial coordinates.



### Exercise 1.2<a name="exercise-12"></a>
```python
def mse(x, y):
    return torch.mean((x - y)**2)
# Vary number of time steps
num_samples =100000
num_steps = [1, 10, 50, 100, 500,1000,5000]

batch_size = 50
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


**Usage**

Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

To use this code, simply run the mse function first.

The main part of the code involves varying the number of time steps and calculating the error of the Monte Carlo simulation for each number of steps. The error of the simulation is then calculated and plotted against the number of time steps.

You also can set different num_samples, num_steps, batch_size(number of samples of x and t), and the distribution of x and t. In this code, the true value of optimal value is obtained by result.value(t,x), if you want to change the parameters in this, you need to call the LQR class and change the input of result variable, which is defined above.

### Exercise 1.2<a name="exercise-12"></b>

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
Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

To use this code, simply run the Python script. The simulation parameters can be adjusted by changing the values of `num_samples_list`, `N`, `batch_size`(number of samples of x and t), `x`, and `t`. The resulting plot shows the mean squared error between the true value of the solution and the estimated value of the solution for each value of `num_samples`.

## Part 2: Supervised learning, checking the NNs are good enough<a name="part-2-supervised-learning-checking-the-nns-are-good-enough"></a>

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

**Usage**
Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

To use this code, run the code of class Net_DGM and the codes for Exercise 1.1 first and then run this code.

The code applies the DGM neural network to approximate the optimal value obtained by the result.value function.

You also can set different batch_size(number of samples of x and t) and the distribution of x_trian and t_train. In this code, the optimal value is obtained by result.value(t,x), if you want to change the parameters in this, you need to call the LQR class and change the input of result variable, which is defined in the second code of Exercise 1.1.


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


**Usage**
Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

To use this code, run the code of class FNN first and the codes for Exercise 1.1 and then run this code.

The code applies the FNN neural network to approximate the optimal action obtained by the lqr.control function.

You also can set different N_data, the terminal time T and the distribution of x_data and t_data. Notice that the terminal time need to be the same in the input of lqr and in the generating of t_data. In this code, the optimal value is obtained by lqr.control(t,x), if you want to change the parameters in this, you need to change the input of lqr variable.


## Part 3: Deep Galerkin approximation for a linear PDE<a name="part-3-deep-galerkin-approximation-for-a-linear-pde"></a>

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

**Usage**


- Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

- Run the code for Exercise 1 and the code of class 'Net_DGM' , 'Net_Layer' in the first code of Exercise 2.1 first and then run all the code above.

You also can set different batch_size(number of samples of x and t), the terminal time T and the distribution of x_3_batch and t_3_batch. If you want to change the parameters in the LQR problem, you need to change corresponding vairables defined in the second code of Exercise 1.1.


## Part 4: Policy iteration with DGM<a name="part-4-policy-iteration-with-dgm"></a>

```python
# Neural network
dim_x = 2
dim_S = 100
sizes = [3, 100, 100, 2]  # input_dim=3 (t, x1, x2), output_dim=2
net_act = FFN(sizes).double()
net_val = Net_DGM(dim_x, dim_S).double()

T = 1
batch_size = 500
t_4_batch  = (torch.rand(batch_size, requires_grad=True) * T).double()
x_4_batch= (torch.rand(batch_size,2, requires_grad=True)*2-1).double()
tx_batch = torch.cat((t_4_batch.unsqueeze(1), x_4_batch), dim=1)
v_true = result.value(t_4_batch.detach(),x_4_batch.unsqueeze(1).detach())
a_true = result.control(t_4_batch.detach(),x_4_batch.unsqueeze(1).detach())


optimizer_val = torch.optim.Adam(net_val.parameters(), lr=0.0001)
optimizer_act = torch.optim.Adam(net_act.parameters(), lr=0.0001) 



loss_fn = nn.MSELoss()

max_iterations = 200

policy_iterations = 20
val_losses = []
act_losses = []
error_act_history=[]
error_value_history=[]
# Policy iteration
for policy_iter in range(policy_iterations):
    # i) Update the value function (theta_val)
    for it in range(max_iterations):
        optimizer_val.zero_grad()
        
        a_act =  net_act(tx_batch)
    
        u_of_tx = net_val(t_4_batch.unsqueeze(1), x_4_batch)
        grad_u_x = get_gradient(u_of_tx,x_4_batch)
        grad_u_t = get_gradient(u_of_tx,t_4_batch)
        grad_u_xx = get_hess(grad_u_x, x_4_batch)
    
        trace = torch.matmul(sigma.view(1, 2, 1).double(), sigma.view(1, 1, 2).double()) @ grad_u_xx.double()
        target_functional = torch.zeros_like(u_of_tx)

        pde = grad_u_t+ 0.5 * ((trace[:,0,0]) + (trace[:,1,1])) + (grad_u_x.unsqueeze(1) @ H @ x_4_batch.unsqueeze(2))[:,0,0] + (grad_u_x.unsqueeze(1) @ M @ a_act.unsqueeze(2))[:,0,0] + ((x_4_batch).unsqueeze(1) @ (C @ x_4_batch.unsqueeze(2)))[:,0,0] + (a_act.unsqueeze(1) @ D @ a_act.unsqueeze(2 ))[:,0,0]

        MSE_functional = loss_fn(torch.reshape(pde,(batch_size,1)), target_functional)
    

    # Compute the terminal condition residual       
    
        x_batch_terminal = x_4_batch
        t_batch_terminal =(torch.ones(batch_size) * T).double()
        u_of_Tx = net_val(t_batch_terminal.unsqueeze(1), x_batch_terminal)
        target_terminal = x_batch_terminal.unsqueeze(1) @ R @ x_batch_terminal.unsqueeze(2)
 
        MSE_terminal = loss_fn(u_of_Tx, target_terminal[:,:,0])
    # Compute total loss
        loss = MSE_functional + MSE_terminal
        loss.backward(retain_graph=True)

    # Backpropagation and optimization
        optimizer_val.step()
        
        val_losses.append(loss.item())
    error_value = loss_fn(v_true,u_of_tx)
    error_value_history.append(error_value.item())    
    # ii) Update the control function (theta_act)
    for it in range(max_iterations):
        optimizer_act.zero_grad()
        a_act =  net_act(tx_batch)
        u_of_tx = net_val(t_4_batch.unsqueeze(1), x_4_batch)
        grad_u_x = get_gradient(u_of_tx,x_4_batch)
        grad_u_t = get_gradient(u_of_tx,t_4_batch)
        grad_u_xx = get_hess(grad_u_x, x_4_batch)
        

        Hami = (grad_u_x.unsqueeze(1) @ H @ x_4_batch.unsqueeze(2))[:,0,0] + (grad_u_x.unsqueeze(1) @ M @ a_act.unsqueeze(2))[:,0,0] + (x_4_batch.unsqueeze(1) @ C @ x_4_batch.unsqueeze(2))[:,0,0] + (a_act.unsqueeze(1) @ D @ a_act.unsqueeze(2))[:,0,0]

        Hamiltonian = (torch.mean(Hami,dim=0 ))
    

    # Compute the terminal condition residual       
    
    # Compute total loss
        loss1 = Hamiltonian
        loss1.backward(retain_graph=True)

    # Backpropagation and optimization
    
        optimizer_act.step()
       
        act_losses.append(loss1.item())
    error_act = loss_fn(a_true,a_act)
    error_act_history.append(error_act.item())

plt.figure(figsize=(15, 8))
plt.plot(val_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Value Function Training Loss')
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(act_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Control Function Training Loss')
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(error_act_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Value Function Training Error')
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(error_value_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Control Function Training Error')
plt.show()
```

**Neural Networks**

Control Function (FFN):
- Input dimensions: 3 (t, x1, x2)
- Output dimensions: 2 (control actions)
- Network architecture: [3, 100, 100, 2]

Value Function (DGM):
- State dimension: 2
- Space dimension: 100

**Training**

The training process includes the following parameters:
- Batch size: 500
- Learning rate for value function: 0.0001
- Learning rate for control function: 0.0001
- Maximum iterations: 200
- Policy iterations: 20
- Loss function: Mean squared error (MSE)

**Usage**

- Ensure you have Python and the necessary libraries installed (PyTorch, NumPy, and Matplotlib).

- Run the code for Exercise 1 first ,the code of function 'get_gradient' and 'get_hess' in Exercise 3, the code of class 'FFN' and class 'Net_DGM' , 'Net_Layer' in the first code of Exercise 2.1 and 2.2 and then run all the code above.

You also can set different batch_size(number of samples of x and t), the terminal time T and the distribution of x_4_batch and t_4_batch. In this code, the true value of optimal value is obtained by result.value(t_4_batch,x_4_batch), the true value of optimal control is obtained by result.control(t_4_batch,x_4_batch). If you want to change the parameters in the corresponding LQR problem, you need to call the LQR class and change the input of result variable, which is defined in the second code of Exercise 1.1.