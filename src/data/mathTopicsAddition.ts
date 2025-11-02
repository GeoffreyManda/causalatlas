// This file contains all the new math topics to be added to theory.ts
// Insert these topics after statistical_inference (line 689) and before empirical_processes (line 691)

export const newMathTopics = `
  // === INTERMEDIATE TIER: Core Applied Methods ===
  
  {
    id: 'generalized_linear_models',
    title: 'Generalized Linear Models',
    tier: 'Intermediate',
    description: 'Exponential families, link functions, logistic/Poisson regression, and deviance',
    content: \`GLMs extend linear regression to non-normal outcomes, crucial for binary treatments and count data in causal inference.\`,
    prerequisites: ['regression_analysis', 'likelihood_theory'],
    learningObjectives: [
      'Understand exponential family distributions',
      'Apply appropriate link functions',
      'Fit logistic and Poisson regression',
      'Use deviance for model comparison'
    ],
    keyDefinitions: [
      { term: 'Exponential family', definition: 'f(y|θ) = exp{[yθ - b(θ)]/a(φ) + c(y,φ)} canonical form' },
      { term: 'Link function', definition: 'g(μ) = η where μ = E[Y] and η = Xβ linear predictor' },
      { term: 'Logistic regression', definition: 'log(p/(1-p)) = Xβ for binary outcomes' },
      { term: 'Deviance', definition: 'D = -2[log L(β̂) - log L(saturated)] measures model fit' }
    ],
    examples: {
      python: \`# Generalized Linear Models
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
n = 200
X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
true_beta = np.array([0.5, 1.0, -0.8])
logit = X @ true_beta
probs = 1 / (1 + np.exp(-logit))
y = np.random.binomial(1, probs)

def neg_log_lik(beta, X, y):
    logit = X @ beta
    return -np.sum(y * logit - np.log(1 + np.exp(logit)))

result = minimize(neg_log_lik, x0=np.zeros(3), args=(X, y), method='BFGS')
print(f"Logistic Regression β̂: {result.x}")
print(f"True β: {true_beta}")
print(f"Odds Ratios: {np.exp(result.x)}")\`,
      r: \`# Generalized Linear Models
set.seed(42)
n <- 200
X1 <- rnorm(n)
X2 <- rnorm(n)
true_beta <- c(0.5, 1.0, -0.8)
logit <- true_beta[1] + true_beta[2]*X1 + true_beta[3]*X2
probs <- 1 / (1 + exp(-logit))
y <- rbinom(n, 1, probs)

fit <- glm(y ~ X1 + X2, family=binomial(link="logit"))
cat("Logistic Regression Results:\\n")
print(summary(fit)$coefficients)
cat("\\nOdds Ratios:\\n")
print(exp(coef(fit)))\`
    },
    references: [
      { authors: 'McCullagh P, Nelder JA', title: 'Generalized Linear Models', year: 1989, doi: '10.1007/978-1-4899-3242-6' }
    ]
  },

  {
    id: 'nonparametric_methods',
    title: 'Nonparametric Methods',
    tier: 'Intermediate',
    description: 'Kernel density estimation, local polynomial regression, bandwidth selection, smoothing',
    content: \`Nonparametric methods allow flexible estimation without strong parametric assumptions, essential for RDD and modern causal inference.\`,
    prerequisites: ['regression_analysis', 'convergence_limit_theorems'],
    learningObjectives: [
      'Estimate densities using kernel methods',
      'Apply local polynomial regression',
      'Select optimal bandwidths',
      'Understand bias-variance tradeoff'
    ],
    keyDefinitions: [
      { term: 'Kernel density estimator', definition: 'f̂(x) = (1/nh)Σ K((x-Xi)/h) where K is kernel, h bandwidth' },
      { term: 'Local linear regression', definition: 'Fit weighted regression in neighborhood of x₀' },
      { term: 'Bandwidth', definition: 'Smoothing parameter h controlling bias-variance tradeoff' },
      { term: 'Cross-validation', definition: 'Data-driven bandwidth selection minimizing prediction error' }
    ],
    examples: {
      python: \`# Nonparametric Methods
import numpy as np

np.random.seed(42)
n = 200
x = np.random.uniform(-3, 3, n)
y = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 2, n)

def gaussian_kernel(u):
    return (1/np.sqrt(2*np.pi)) * np.exp(-u**2/2)

def kernel_density(x_eval, data, h):
    n = len(data)
    density = np.zeros_like(x_eval)
    for i, x_val in enumerate(x_eval):
        u = (x_val - data) / h
        density[i] = (1/(n*h)) * np.sum(gaussian_kernel(u))
    return density

x_grid = np.linspace(-3, 3, 50)
density_est = kernel_density(x_grid, x, bandwidth=0.3)
print(f"Kernel Density at x=0: {kernel_density(np.array([0]), x, 0.3)[0]:.4f}")
print(f"Min density: {density_est.min():.4f}")
print(f"Max density: {density_est.max():.4f}")\`,
      r: \`# Nonparametric Methods
set.seed(42)
n <- 200
x <- runif(n, -3, 3)
y <- 2 * x^2 - 3 * x + 1 + rnorm(n, 0, 2)

dens <- density(x, bw=0.3, kernel="gaussian")
cat("Kernel Density Estimation:\\n")
cat("Bandwidth:", dens$bw, "\\n")
cat("Density at x=0:", approx(dens$x, dens$y, xout=0)$y, "\\n")

fit_loess <- loess(y ~ x, span=0.3, degree=1)
cat("\\nLocal Linear Regression:\\n")
cat("Span:", fit_loess$pars$span, "\\n")\`
    },
    references: [
      { authors: 'Wasserman L', title: 'All of Nonparametric Statistics', year: 2006, doi: '10.1007/0-387-30623-4' }
    ]
  },

  {
    id: 'bootstrap_resampling',
    title: 'Bootstrap & Resampling Methods',
    tier: 'Intermediate',
    description: 'Bootstrap theory, block bootstrap, permutation tests, and subsampling',
    content: \`Bootstrap provides distribution-free inference, crucial when asymptotic approximations fail or for complex estimands.\`,
    prerequisites: ['statistical_inference', 'asymptotic_theory'],
    learningObjectives: [
      'Apply nonparametric bootstrap',
      'Use block bootstrap for time series',
      'Conduct permutation tests',
      'Construct bootstrap confidence intervals'
    ],
    keyDefinitions: [
      { term: 'Bootstrap', definition: 'Resample with replacement to approximate sampling distribution' },
      { term: 'Bootstrap percentile CI', definition: '[θ̂*α/2, θ̂*1-α/2] from bootstrap quantiles' },
      { term: 'Block bootstrap', definition: 'Resample blocks to preserve time dependence' },
      { term: 'Permutation test', definition: 'Randomly permute labels to test null hypothesis' }
    ],
    examples: {
      python: \`# Bootstrap & Resampling
import numpy as np

np.random.seed(42)
n = 50
data = np.random.exponential(scale=2, size=n)

theta_hat = data.mean()
print(f"Point estimate: {theta_hat:.3f}")

# Nonparametric bootstrap
B = 1000
bootstrap_estimates = np.zeros(B)
for b in range(B):
    resample = np.random.choice(data, size=n, replace=True)
    bootstrap_estimates[b] = resample.mean()

se_boot = bootstrap_estimates.std()
ci_lower = np.percentile(bootstrap_estimates, 2.5)
ci_upper = np.percentile(bootstrap_estimates, 97.5)

print(f"\\nBootstrap SE: {se_boot:.3f}")
print(f"95% Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"Bootstrap bias: {bootstrap_estimates.mean() - theta_hat:.4f}")\`,
      r: \`# Bootstrap & Resampling
set.seed(42)
n <- 50
data <- rexp(n, rate=1/2)

theta_hat <- mean(data)
cat("Point estimate:", theta_hat, "\\n")

# Nonparametric bootstrap
B <- 1000
bootstrap_estimates <- replicate(B, {
  resample <- sample(data, size=n, replace=TRUE)
  mean(resample)
})

se_boot <- sd(bootstrap_estimates)
ci <- quantile(bootstrap_estimates, c(0.025, 0.975))

cat("\\nBootstrap SE:", se_boot, "\\n")
cat("95% Bootstrap CI: [", ci[1], ",", ci[2], "]\\n")
cat("Bootstrap bias:", mean(bootstrap_estimates) - theta_hat, "\\n")\`
    },
    references: [
      { authors: 'Efron B, Tibshirani RJ', title: 'An Introduction to the Bootstrap', year: 1994, doi: '10.1201/9780429246593' }
    ]
  },

  {
    id: 'bayesian_statistics',
    title: 'Bayesian Statistics',
    tier: 'Advanced',
    description: 'Prior/posterior distributions, credible intervals, hierarchical models, MCMC basics',
    content: \`Bayesian methods provide an alternative paradigm for causal inference, incorporating prior knowledge and enabling coherent uncertainty quantification.\`,
    prerequisites: ['likelihood_theory', 'multivariate_distributions'],
    learningObjectives: [
      'Specify prior distributions',
      'Compute posterior distributions',
      'Construct credible intervals',
      'Understand Bayesian vs frequentist inference'
    ],
    keyDefinitions: [
      { term: 'Prior', definition: 'π(θ) represents belief about θ before observing data' },
      { term: 'Posterior', definition: 'π(θ|y) ∝ L(y|θ)π(θ) by Bayes theorem' },
      { term: 'Credible interval', definition: '[L, U] with P(θ ∈ [L,U]|data) = 1-α' },
      { term: 'Conjugate prior', definition: 'Prior and posterior in same distributional family' }
    },
    examples: {
      python: \`# Bayesian Statistics
import numpy as np
from scipy import stats

np.random.seed(42)
n = 30
true_mu = 5.0
data = np.random.normal(true_mu, 2, n)

# Prior: μ ~ N(0, 10²)
prior_mu = 0
prior_var = 10**2

# Likelihood: data ~ N(μ, 2²)
sigma = 2.0
data_mean = data.mean()

# Posterior: μ|data ~ N(μ_post, σ²_post)
precision_prior = 1/prior_var
precision_data = n/(sigma**2)
posterior_precision = precision_prior + precision_data
posterior_var = 1/posterior_precision
posterior_mu = posterior_var * (precision_prior*prior_mu + precision_data*data_mean)

print(f"Bayesian Inference for Normal Mean:")
print(f"Prior: N({prior_mu}, {prior_var})")
print(f"Data mean: {data_mean:.3f}")
print(f"\\nPosterior: N({posterior_mu:.3f}, {posterior_var:.3f})")

# 95% Credible interval
ci_lower = posterior_mu - 1.96*np.sqrt(posterior_var)
ci_upper = posterior_mu + 1.96*np.sqrt(posterior_var)
print(f"95% Credible Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Posterior probability
prob_positive = 1 - stats.norm.cdf(0, posterior_mu, np.sqrt(posterior_var))
print(f"\\nP(μ > 0 | data) = {prob_positive:.4f}")\`,
      r: \`# Bayesian Statistics
set.seed(42)
n <- 30
true_mu <- 5.0
data <- rnorm(n, true_mu, 2)

# Prior: μ ~ N(0, 10²)
prior_mu <- 0
prior_var <- 10^2

# Likelihood: data ~ N(μ, 2²)
sigma <- 2.0
data_mean <- mean(data)

# Posterior: μ|data ~ N(μ_post, σ²_post)
precision_prior <- 1/prior_var
precision_data <- n/(sigma^2)
posterior_precision <- precision_prior + precision_data
posterior_var <- 1/posterior_precision
posterior_mu <- posterior_var * (precision_prior*prior_mu + precision_data*data_mean)

cat("Bayesian Inference for Normal Mean:\\n")
cat("Prior: N(", prior_mu, ",", prior_var, ")\\n")
cat("Data mean:", data_mean, "\\n")
cat("\\nPosterior: N(", round(posterior_mu, 3), ",", round(posterior_var, 3), ")\\n")

# 95% Credible interval
ci_lower <- posterior_mu - 1.96*sqrt(posterior_var)
ci_upper <- posterior_mu + 1.96*sqrt(posterior_var)
cat("95% Credible Interval: [", round(ci_lower, 3), ",", round(ci_upper, 3), "]\\n")

# Posterior probability
prob_positive <- 1 - pnorm(0, posterior_mu, sqrt(posterior_var))
cat("\\nP(μ > 0 | data) =", round(prob_positive, 4), "\\n")\`
    },
    references: [
      { authors: 'Gelman A et al', title: 'Bayesian Data Analysis', year: 2013, doi: '10.1201/b16018' }
    ]
  },

  {
    id: 'high_dimensional_statistics',
    title: 'High-Dimensional Statistics',
    tier: 'Advanced',
    description: 'Lasso, ridge regression, elastic net, variable selection, and sparsity',
    content: \`High-dimensional methods handle modern datasets where number of variables exceeds sample size, crucial for machine learning in causal inference.\`,
    prerequisites: ['regression_analysis', 'asymptotic_theory'],
    learningObjectives: [
      'Apply Lasso for variable selection',
      'Use ridge regression for regularization',
      'Understand bias-variance tradeoff in high dimensions',
      'Select tuning parameters via cross-validation'
    ],
    keyDefinitions: [
      { term: 'Lasso', definition: 'min ||y - Xβ||² + λ||β||₁ with L1 penalty for sparsity' },
      { term: 'Ridge regression', definition: 'min ||y - Xβ||² + λ||β||² with L2 penalty for shrinkage' },
      { term: 'Elastic net', definition: 'Combines L1 and L2 penalties: α||β||₁ + (1-α)||β||²' },
      { term: 'Cross-validation', definition: 'Choose λ minimizing prediction error on held-out data' }
    ],
    examples: {
      python: \`# High-Dimensional Statistics
import numpy as np

np.random.seed(42)
n = 100
p = 50  # p < n but illustrates concepts
X = np.random.randn(n, p)

# True model: only 5 variables matter
true_beta = np.zeros(p)
true_beta[:5] = [3, -2, 1.5, -1, 2.5]
y = X @ true_beta + np.random.randn(n)

# Ridge regression (closed form)
lambda_ridge = 1.0
I = np.eye(p)
beta_ridge = np.linalg.solve(X.T @ X + lambda_ridge * I, X.T @ y)

print(f"Ridge Regression (λ={lambda_ridge}):")
print(f"Number of nonzero coefficients: {(np.abs(beta_ridge) > 0.01).sum()}")
print(f"L2 norm: {np.linalg.norm(beta_ridge):.3f}")
print(f"First 5 coefficients: {beta_ridge[:5]}")

# Simple Lasso via coordinate descent (illustration)
def soft_threshold(x, lambda_val):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

# Initialize
beta_lasso = np.zeros(p)
lambda_lasso = 0.5
max_iter = 100

for iteration in range(max_iter):
    for j in range(p):
        # Partial residual
        r = y - X @ beta_lasso + X[:, j] * beta_lasso[j]
        # Update j-th coefficient
        beta_lasso[j] = soft_threshold(X[:, j] @ r / n, lambda_lasso) / (X[:, j] @ X[:, j] / n)

print(f"\\nLasso (λ={lambda_lasso}):")
print(f"Number of nonzero coefficients: {(np.abs(beta_lasso) > 0.01).sum()}")
print(f"L1 norm: {np.linalg.norm(beta_lasso, 1):.3f}")
print(f"First 5 coefficients: {beta_lasso[:5]}")\`,
      r: \`# High-Dimensional Statistics
set.seed(42)
n <- 100
p <- 50
X <- matrix(rnorm(n*p), n, p)

# True model: only 5 variables matter
true_beta <- c(3, -2, 1.5, -1, 2.5, rep(0, p-5))
y <- X %*% true_beta + rnorm(n)

# Ridge regression
lambda_ridge <- 1.0
I <- diag(p)
beta_ridge <- solve(t(X) %*% X + lambda_ridge * I) %*% t(X) %*% y

cat("Ridge Regression (λ=", lambda_ridge, "):\\n", sep="")
cat("Number of nonzero coefficients:", sum(abs(beta_ridge) > 0.01), "\\n")
cat("L2 norm:", round(sqrt(sum(beta_ridge^2)), 3), "\\n")
cat("First 5 coefficients:", beta_ridge[1:5], "\\n")

# Lasso using glmnet package
if (requireNamespace("glmnet", quietly=TRUE)) {
  library(glmnet)
  fit_lasso <- glmnet(X, y, alpha=1, lambda=0.5)
  beta_lasso <- as.vector(coef(fit_lasso))[-1]  # Remove intercept
  
  cat("\\nLasso (λ=0.5):\\n")
  cat("Number of nonzero coefficients:", sum(abs(beta_lasso) > 0.01), "\\n")
  cat("L1 norm:", round(sum(abs(beta_lasso)), 3), "\\n")
  cat("First 5 coefficients:", beta_lasso[1:5], "\\n")
} else {
  cat("\\nInstall 'glmnet' package for Lasso\\n")
}\`
    },
    references: [
      { authors: 'Hastie T, Tibshirani R, Wainwright M', title: 'Statistical Learning with Sparsity', year: 2015, doi: '10.1201/b18401' }
    ]
  },
`;
