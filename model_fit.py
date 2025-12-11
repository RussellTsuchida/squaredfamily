import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIAS_SCALE = np.sqrt(0.2)

class BiasReparam(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.hstack((x, 
                BIAS_SCALE*torch.ones(x.shape[0], 1, device=device)))

class Psi(torch.nn.Module):
    def __init__(self, hidden_layer_size, activation='relu'):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self._init_activation(activation)

        # initialise hidden layer
        self.feature_layer = torch.nn.Sequential(\
                        torch.nn.Linear(2, self.hidden_layer_size-1, device=device,
                                bias=False),
                self.activation,
                BiasReparam())

        for param in self.feature_layer.parameters():
            param.requires_grad = False

    def _init_activation(self, activation):
        self.activation_str = activation
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        if activation == 'gelu':
            self.activation = torch.nn.GELU(approximate='none')

    def forward(self, x):
        return self.feature_layer(x)

    def compute_kernel(self, base_variance):
        if self.activation_str == 'relu':
            return self.relu_kernel(base_variance)
        if self.activation_str == 'gelu':
            return self.gelu_kernel(base_variance)

    def relu_kernel(self, base_variance):
        wb = self.feature_layer[0].weight

        inner_products = wb @ wb.T
        norms = torch.sqrt(torch.diag(inner_products)).reshape((-1,1))
        norms_outer = norms @ norms.T

        cos = torch.clip(inner_products / norms_outer, -1, 1)
        sin = torch.clip(torch.sqrt(1 - cos**2), 0, 1)
        theta = torch.arccos(cos)
        
        # Top block of kernel (minus 1 row and 1 column)
        sub_kernel = norms @ norms.T*base_variance \
                / (2*np.pi) * (sin + (np.pi - theta) * cos)
        self.kernel = torch.zeros(sub_kernel.shape[0]+1, sub_kernel.shape[0]+1,
                                  device = device)

        self.kernel[:-1, :-1] = sub_kernel
        
        # Fill in the last row and column
        # This is the expected value of the ReLU
        row = np.sqrt(2/np.pi)/2 * \
                torch.linalg.norm(wb,dim=1).reshape((-1,1))*\
                np.sqrt(base_variance)
        self.kernel[:-1, -1] = row.reshape((-1,))*BIAS_SCALE
        self.kernel[-1, :-1] = row.reshape((-1,))*BIAS_SCALE
        self.kernel[-1,-1] = BIAS_SCALE**2

        # Check for strict PD
        eigs = torch.linalg.eigvals(self.kernel)
        print(torch.min(torch.real(eigs)))
        return self.kernel

    def gelu_kernel(self, base_variance):
        _eps = 0#1e-12
        wb = self.feature_layer[0].weight  # (n, d)
        inner = wb @ wb.T                   # (n, n) unscaled dot products
        norms_unscaled = torch.sqrt(torch.diag(inner))   # (n,)
        norms_unscaled = norms_unscaled.clamp(min=_eps)
        norms = norms_unscaled.reshape(-1, 1)            # (n,1)
        norms_outer = norms @ norms.T                    # (n,n)

        # cosine must be unscaled angle cosine
        cos = torch.clamp(inner / (norms_outer + _eps), -1.0, 1.0)
        sin = torch.sqrt(torch.clamp(1.0 - cos**2, min=0.0))

        # s1, s2 are sqrt(v_i) = sqrt(base_variance) * ||w_i||
        s1 = norms * torch.sqrt(torch.tensor(base_variance, device=wb.device, dtype=norms.dtype))
        s2 = s1.T
        sprod = s1 * s2       # equals sqrt(v_i) * sqrt(v_j)
        # denom for any division by s1*s2:
        denom_sprod = sprod + _eps

        # A = 1 + s1^2 + s2^2 + s1^2 s2^2 sin^2(theta)
        A = 1.0 + s1**2 + s2**2 + (s1**2) * (s2**2) * (sin**2)
        sqrtA = torch.sqrt(A.clamp(min=_eps))

        # arctan argument = cos * s1 * s2 / sqrtA
        atan_arg = (cos * sprod) / sqrtA
        atan_term = torch.atan(atan_arg)

        # B block
        cos2 = 2.0 * (cos**2) - 1.0
        B = 0.5 * (cos2 + 3.0) + s1**2 + s2**2 + (s1**2) * (s2**2) * (sin**2)

        # R term
        R = (s1**2 * s2**2) * B
        denom_R = (1.0 + s1**2) * (1.0 + s2**2) * sqrtA + _eps
        R = R / denom_R
        R = R / (2.0 * torch.pi)

        # L term
        L = 0.25 * (sprod * cos)

        # T term: (cos / (s1 s2)) * atan(...)
        T = (cos * sprod) * atan_term/(2*torch.pi)

        K_block = L + R + T

        n = K_block.shape[0]
        K = torch.zeros(n + 1, n + 1, device=wb.device, dtype=K_block.dtype)
        K[:-1, :-1] = K_block

        # Fill bias cross-terms: E[GELU] = v / sqrt(2*pi (1+v)), with v = base_variance * ||w||^2
        v = base_variance * torch.sum(wb * wb, dim=1)   # shape (n,)
        row = v / torch.sqrt(2.0 * torch.pi * (1.0 + v))

        K[:-1, -1] = row.reshape(-1) * BIAS_SCALE
        K[-1, :-1] = row.reshape(-1) * BIAS_SCALE
        K[-1, -1] = BIAS_SCALE**2

        #eigs = torch.linalg.eigvalsh(K)
        #print(torch.min(eigs).item())
        return K

class SquaredFamily(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden_layer_size = 3
        self.base_variance = 1.
        self.proposal_variance = 2.

        self.psi = Psi(self.hidden_layer_size, 'gelu')
        
        self.param_layer = torch.nn.Linear(self.hidden_layer_size, 1, bias=False, 
                                            device=device)

        self.kernel = self.psi.compute_kernel(self.base_variance)
        print(self.kernel)
        print(torch.linalg.pinv(self.kernel))

    def normalise_param_layer(self):
        with torch.no_grad():
            param = self.param_layer.weight
            param = param / torch.sqrt(self.normalising_constant())
            param = param * torch.sign(param[0,-1])
            self.param_layer.weight = torch.nn.Parameter(param)


    def reinitialise_param_layer(self, param=None):
        with torch.no_grad():
            self.param_layer = torch.nn.Linear(self.hidden_layer_size, 1, bias=False, 
                                                device=device)
            if not (param is None):
                self.param_layer.weight = torch.nn.Parameter(param)

        for param in self.psi.feature_layer.parameters():
            param.requires_grad = False

    def forward(self, x, dimension_augmentation = False, a = None):
        #psi = self.feature_layer(x)
        psi = self.psi(x)
        lin = self.param_layer(psi)

        den = self.normalising_constant()

        base_density=-1/(2*self.base_variance)*torch.linalg.norm(x,dim=1)**2+\
                -(x.shape[1]/2*np.log(2*np.pi)+\
                             x.shape[1]/2*np.log(self.base_variance))

        if dimension_augmentation:
            assert not (a is None)
            mean = torch.log(den)
            aug_density = (-(a - mean)**2)/2 - 0.5*np.log(2*np.pi)
        else:
            aug_density = 0
        
        #if dimension_augmentation: 
        #    return aug_density

        return torch.log(lin**2)-torch.log(den)+base_density.reshape((-1,1))+\
                aug_density



    def normalising_constant(self):
        return torch.sum(
                (self.param_layer.weight.T @ self.param_layer.weight)*
                self.kernel)

    def _sample(self, num_samples):
        with torch.no_grad():
            # Compute the bound on the likelihood ratio
            s = self.base_variance**-1 - self.proposal_variance**-1
            _, sval, _ = torch.linalg.svd(self.psi.feature_layer[0].weight)
            w = sval[0]

            den = self.normalising_constant()

            if w**2 <= s/2:
                bound = 1
            else:
                bound = torch.exp(-(2*w**2 - s)/(2*w**2))*2 * w**2 / s
            bound = bound * self.proposal_variance/self.base_variance*\
                    torch.linalg.norm(self.param_layer.weight) / den

            proposal_samples = torch.normal(torch.zeros(num_samples, 2,
                                                        device=device),
                                            np.sqrt(self.proposal_variance))
            proposal_density =\
            -1/(2*self.proposal_variance)*torch.linalg.norm(proposal_samples,dim=1)**2+\
            -(proposal_samples.shape[1]/2*np.log(2*np.pi)+\
            proposal_samples.shape[1]/2*np.log(self.proposal_variance))

            log_likelihood_ratio = self(proposal_samples)\
                - proposal_density.reshape((-1,1))

            uniform = torch.rand(num_samples, device=device).reshape((-1,1))

            idx = torch.nonzero(torch.log(uniform) < log_likelihood_ratio - torch.log(bound))

            samples = proposal_samples[idx[:,0], :]
            return samples

    def sample(self, num_samples):
        with torch.no_grad():
            batch_size = 10000
            samples = torch.empty(0, 2, device=device)
            while samples.shape[0] < num_samples:
                samples = torch.vstack((samples, self._sample(batch_size)))

            return samples[:num_samples, :]

    def rayleigh_estimate(self, data):         
        with torch.no_grad():
            psi = self.psi(data).unsqueeze(2)

            den = self.normalising_constant()

            base_density=(-1/(2*self.base_variance)*torch.linalg.norm(data,dim=1)**2+\
                    -(data.shape[1]/2*np.log(2*np.pi)+\
                    data.shape[1]/2*np.log(self.base_variance))).\
                    reshape((-1,1,1)).tile((1,psi.shape[1],psi.shape[1]))
            
            psi_psit = (torch.bmm(psi, torch.transpose(psi, 1, 2))) 

            mid_fac = torch.mean(torch.exp(base_density) * psi_psit, dim=0)
            L = torch.linalg.cholesky(self.kernel)
            mat = torch.linalg.inv(L) @ mid_fac @ torch.linalg.inv(L).T
            _, vmax = torch.lobpcg(mat)

            param = torch.linalg.inv(L).T @ vmax
            param = param / torch.sqrt(self.normalising_constant())
            #param = param * 1
            self.param_layer.weight = torch.nn.Parameter(param.T)


if __name__ == '__main__':
    import scipy.integrate as integrate
    NUM_TRIALS = 500        # Number of times to attempt to solve MLE
    NUM_SAMPLES = 1000      # Number of samples for fitting MLE
    num_epochs = 500        # Number of training epochs

    torch.manual_seed(2)
    density = SquaredFamily() 
    density.normalise_param_layer()
    true_params = density.param_layer.weight
    samples = density.sample(NUM_SAMPLES)

    def plot_density(fname, samples=None):
        plt.figure(figsize=(8, 6))
        if not (samples is None):
            xmin = torch.min(samples[:,0]).cpu().numpy()
            xmax = torch.max(samples[:,0]).cpu().numpy()

            ymin = torch.min(samples[:,1]).cpu().numpy()
            ymax = torch.max(samples[:,1]).cpu().numpy()

            samples = samples.cpu().numpy()

        else:
            xmin = -3; xmax = 3; ymin = -3; ymax = 3

        # Plot density heatmap
        x_range = np.linspace(xmin, xmax, 100)
        y_range = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x_range, y_range)
        grid = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, 
                            dtype=torch.float32, device=device)

        with torch.no_grad():
            Z = density(grid).cpu().numpy().reshape(100, 100)

        plt.contourf(X, Y, np.exp(Z), levels=50, cmap='viridis')
        plt.colorbar(label='Density')

        if not (samples is None):
            plt.scatter(samples[:, 0], samples[:, 1], color='red', s=5,
                        alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(fname)
        plt.close()

    plot_density('samples.png', samples)

    # Verify that density integrates to 1 (takes a few seconds)
    if False:
        pdf = lambda x1, x2: torch.exp(density(torch.tensor([[x1, x2]],
                                                            device=device))).\
                    cpu().detach().numpy()
        print('Numerical integration of the PDF gives the following:')
        integral = integrate.dblquad(pdf, -np.inf, np.inf, lambda x: -np.inf, 
                          lambda x: np.inf, epsabs = 1e-4, epsrel=1e-4)
        print(integral)

    # Now repeat the MLE many times with different initial theta and
    # different data
    mle_params = torch.zeros(NUM_TRIALS+1, 
                             density.hidden_layer_size+1, device=device)
    for t in range(1,NUM_TRIALS+1):
        density.reinitialise_param_layer(true_params)
        x_train = density.sample(NUM_SAMPLES)
        a_train = torch.normal(torch.zeros(\
                x_train.shape[0],1, device=device), 1)
        density.rayleigh_estimate(x_train)
        density.train()

        # Convert to torch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32, device=device)

        # Set up the optimizer
        optimizer = torch.optim.Adam(density.param_layer.parameters(), lr=0.1)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Compute training loss 
            log_p = density(x_train, dimension_augmentation=True,
                        a = a_train)
            train_loss = -torch.mean(log_p)
            train_loss.backward()
            optimizer.step()
        
        print("Completed " + str(t) + " out of " + str(NUM_TRIALS) + " MLE fits.")
        #print(density.param_layer.weight[0].detach().cpu().numpy())
        #print(train_loss.item())
        #print(density.normalising_constant())
        #plot_density(str(t) + '.png', x_train)
        mle_params[t,:-1] = density.param_layer.weight #\
        #/ torch.sqrt(density.normalising_constant())

        mle_params[t,-1] = train_loss.item()
    
    density.eval()
    mle_params[0,:-1] = true_params

    np.save('params.npy', mle_params.cpu().detach().numpy())
    np.save('kernel.npy', density.kernel.cpu().detach().numpy())



