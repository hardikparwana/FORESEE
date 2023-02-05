import gpytorch
import torch

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood)

def train_gp(models, likelihoods, train_x, train_y, training_iterations = 50):
    # Find optimal model hyperparameters
    print("GP training start")
    
    optimizers = []
    mlls = []
    for i in range(4):
    
        models[i].train()
        likelihoods[i].train()
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=0.1))  # Includes GaussianLikelihood parameters
        
        # "Loss" for GPs - the marginal log likelihood
        mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihoods[i], models[i]))

    for i in range(training_iterations):
        
        for j in range(4):
        
            optimizers[j].zero_grad()
            output = models[j](train_x)
            loss = -mlls[j](output, train_y[:,j])
            loss.backward()
            optimizers[j].step()
   
   
    # Set into eval mode
    for i in range(4):
        models[i].eval()
        likelihoods[i].eval()



class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean1_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=1
        )
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernel() *  gpytorch.kernels.LinearKernel() *  gpytorch.kernels.LinearKernel(), num_tasks=num_tasks, rank=1
        # )
        self.covar1_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel() *  gpytorch.kernels.LinearKernel(), num_tasks=1, rank=1
        )
        self.mean2_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=1
        )
        self.covar2_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel() *  gpytorch.kernels.LinearKernel(), num_tasks=1, rank=1
        )
        self.mean3_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=1
        )
        self.covar3_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel() *  gpytorch.kernels.LinearKernel(), num_tasks=1, rank=1
        )
        self.mean4_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=1
        )
        self.covar4_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel() *  gpytorch.kernels.LinearKernel(), num_tasks=1, rank=1
        )

    def forward(self, x):
        mean1_x = self.mean1_module(x)
        covar1_x = self.covar1_module(x)
        mean2_x = self.mean2_module(x)
        covar2_x = self.covar2_module(x)
        mean3_x = self.mean3_module(x)
        covar3_x = self.covar3_module(x)
        mean4_x = self.mean4_module(x)
        covar4_x = self.covar4_module(x)
        mean_x = torch.cat((mean1_x, mean2_x, mean3_x, mean4_x), dim=1)
        covar_x = torch.diag( torch.cat((covar1_x, covar2_x, covar3_x, covar4_x)) )
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)
# model2 = MultitaskGPModel(train_x, train_y, likelihood)
# predictions = likelihood2(model2(test_x))
# mean = predictions.mean
# lower, upper = predictions.confidence_region()
