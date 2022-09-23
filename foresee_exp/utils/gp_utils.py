import gpytorch
import torch

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel() *  gpytorch.kernels.LinearKernel() *  gpytorch.kernels.LinearKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)
# model2 = MultitaskGPModel(train_x, train_y, likelihood)
# predictions = likelihood2(model2(test_x))
# mean = predictions.mean
# lower, upper = predictions.confidence_region()

def train_gp(model, likelihood, train_x, train_y, training_iterations = 50):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # print(f"train gp: x:{ train_x }, y:{ train_y }")
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
    #     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss2.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()