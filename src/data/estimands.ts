export type Framework = 'PotentialOutcomes' | 'SCM' | 'PrincipalStratification' | 'ProximalNegativeControl' | 'BayesianDecision';
export type Design = 'RCT_Parallel' | 'Cluster_RCT' | 'Stepped_Wedge' | 'Factorial' | 'Encouragement' | 'Two_Stage' | 'Cohort' | 'Case_Control' | 'Cross_Sectional' | 'Case_Cohort' | 'SCCS' | 'Case_Crossover' | 'Regression_Discontinuity' | 'Natural_Experiment' | 'Target_Trial_Emulation' | 'Survey_Multistage' | 'Two_Phase' | 'Transport_Frame';
export type EstimandFamily = 'PopulationEffects' | 'DistributionalQuantile' | 'LongitudinalDynamic' | 'InstrumentalLocal' | 'MediationPathSpecific' | 'InterferenceSpillovers' | 'SurvivalTimeToEvent' | 'PartialIDSensitivity' | 'MissingnessMeasurementError' | 'ProximalBridges' | 'TransportExternalValidity' | 'PolicyValueRL' | 'DeepRepresentation';
export type Tier = 'Basic' | 'Intermediate' | 'Advanced' | 'Frontier';
export type DiscoveryStatus = 'identifiable' | 'partially_identifiable' | 'non_identifiable';
export type EIFStatus = 'available' | 'unknown' | 'non_pathwise';

export interface Reference {
  authors: string;
  title: string;
  year: number;
  doi: string;
}

export interface Estimand {
  id: string;
  short_name: string;
  framework: Framework;
  design: Design;
  estimand_family: EstimandFamily;
  tier: Tier;
  definition_tex: string;
  assumptions: string[];
  identification_formula_tex: string;
  estimators: string[];
  discovery_status: DiscoveryStatus;
  eif_status: EIFStatus;
  references: Reference[];
  examples: {
    python: string;
    r: string;
  };
}

export const estimandsData: Estimand[] = [
  
  {
    id: 'ate',
    short_name: 'Average Treatment Effect (ATE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\psi = E[Y^1 - Y^0]',
    assumptions: ['SUTVA', 'Consistency', 'Exchangeability', 'Positivity'],
    identification_formula_tex: 'E_X[E[Y|A=1,X] - E[Y|A=0,X]]',
    estimators: ['G-computation', 'IPW', 'AIPW', 'TMLE'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Robins JM', title: 'A new approach to causal inference', year: 1986, doi: '10.2307/2289144' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,3))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 2*A + X[:,0] + np.random.normal(size=n)
ps = LogisticRegression(max_iter=300).fit(X,A).predict_proba(X)[:,1]
mu1 = LinearRegression().fit(X[A==1], Y[A==1]).predict(X)
mu0 = LinearRegression().fit(X[A==0], Y[A==0]).predict(X)
phi = A/ps*(Y-mu1) - (1-A)/(1-ps)*(Y-mu0) + (mu1-mu0)
print(f"ATE: {phi.mean():.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*3), n, 3)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 2*A + X[,1] + rnorm(n)
ps <- glm(A ~ X, family=binomial)$fitted
w <- ifelse(A==1, 1/ps, 1/(1-ps))
psi <- mean(w*Y*A) - mean(w*Y*(1-A))
cat("ATE:", round(psi, 3), "\\n")`
    }
  },

  {
    id: 'att',
    short_name: 'Average Treatment Effect on Treated (ATT)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'E[Y^1 - Y^0 \\mid A=1]',
    assumptions: ['SUTVA', 'Consistency', 'Conditional exchangeability', 'Positivity'],
    identification_formula_tex: 'E[Y \\mid A=1] - E_X[E[Y \\mid A=0,X] \\mid A=1]',
    estimators: ['IPW for ATT', 'Matching', 'DR-ATT'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Lunceford JK, Davidian M', title: 'Stratification and weighting', year: 2004, doi: '10.1002/sim.1903' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 1.5*A + X[:,0] + np.random.normal(size=n)
ps = LogisticRegression(max_iter=300).fit(X,A).predict_proba(X)[:,1]
att = Y[A==1].mean() - np.sum(Y[A==0]*ps[A==0]/(1-ps[A==0]))/np.sum(ps[A==0]/(1-ps[A==0]))
print(f"ATT: {att:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 1.5*A + X[,1] + rnorm(n)
ps <- glm(A ~ X, family=binomial)$fitted
att <- mean(Y[A==1]) - sum(Y[A==0]*ps[A==0]/(1-ps[A==0]))/sum(ps[A==0]/(1-ps[A==0]))
cat("ATT:", round(att, 3), "\\n")`
    }
  },

  {
    id: 'cate',
    short_name: 'Conditional ATE (CATE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: '\\tau(x) = E[Y^1 - Y^0 \\mid X=x]',
    assumptions: ['SUTVA', 'Consistency', 'Conditional exchangeability', 'Positivity'],
    identification_formula_tex: 'E[Y \\mid A=1, X=x] - E[Y \\mid A=0, X=x]',
    estimators: ['T-learner', 'S-learner', 'X-learner', 'R-learner', 'Causal forests'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Künzel SR et al', title: 'Metalearners for HTE', year: 2019, doi: '10.1073/pnas.1804597116' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.ensemble import RandomForestRegressor
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,3))
A = np.random.binomial(1, 0.5, n)
tau_true = 1 + 0.5*X[:,0]
Y = tau_true*A + X[:,0] + np.random.normal(size=n)
mu1 = RandomForestRegressor(n_estimators=50, random_state=0).fit(X[A==1], Y[A==1])
mu0 = RandomForestRegressor(n_estimators=50, random_state=0).fit(X[A==0], Y[A==0])
cate = mu1.predict(X) - mu0.predict(X)
print(f"CATE mean: {cate.mean():.3f}, std: {cate.std():.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*3), n, 3)
A <- rbinom(n, 1, 0.5)
tau_true <- 1 + 0.5*X[,1]
Y <- tau_true*A + X[,1] + rnorm(n)
mu1 <- lm(Y ~ X, subset=A==1)
mu0 <- lm(Y ~ X, subset=A==0)
cate <- predict(mu1, newdata=data.frame(X)) - predict(mu0, newdata=data.frame(X))
cat("CATE mean:", round(mean(cate), 3), "\\n")`
    }
  },

  {
    id: 'qte',
    short_name: 'Quantile Treatment Effect (QTE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DistributionalQuantile',
    tier: 'Advanced',
    definition_tex: '\\Delta_\\tau = Q_{Y^1}(\\tau) - Q_{Y^0}(\\tau)',
    assumptions: ['SUTVA', 'Consistency', 'Exchangeability', 'Rank invariance'],
    identification_formula_tex: 'Q_{Y|A=1}(\\tau) - Q_{Y|A=0}(\\tau)',
    estimators: ['Sample quantiles', 'Distributional regression', 'IF-based'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Firpo S', title: 'Semiparametric QTE estimation', year: 2007, doi: '10.1111/j.1468-0262.2007.00738.x' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 3000
A = np.random.binomial(1, 0.5, n)
Y = 2*A + np.random.normal(0, 1+0.5*A, n)
tau = 0.75
qte = np.quantile(Y[A==1], tau) - np.quantile(Y[A==0], tau)
print(f"QTE at tau={tau}: {qte:.3f}")`,
      r: `set.seed(20251111)
n <- 3000
A <- rbinom(n, 1, 0.5)
Y <- 2*A + rnorm(n, 0, 1+0.5*A)
tau <- 0.75
qte <- quantile(Y[A==1], tau) - quantile(Y[A==0], tau)
cat("QTE:", round(qte, 3), "\\n")`
    }
  },

  {
    id: 'msm',
    short_name: 'Marginal Structural Model (MSM)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'LongitudinalDynamic',
    tier: 'Intermediate',
    definition_tex: 'E[Y^{\\bar{a}}] = m(\\bar{a}; \\beta)',
    assumptions: ['Sequential exchangeability', 'Positivity', 'Consistency'],
    identification_formula_tex: '\\sum_l E[Y \\mid \\bar{A}=\\bar{a}, \\bar{L}=\\bar{l}] P(\\bar{l})',
    estimators: ['IPW with stabilized weights', 'TMLE', 'G-computation'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Robins JM et al', title: 'Marginal structural models', year: 2000, doi: '10.1097/00001648-200009000-00011' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 3000
L0 = np.random.normal(size=n)
A0 = np.random.binomial(1, 1/(1+np.exp(-0.5*L0)))
L1 = 0.4*A0 + L0 + np.random.normal(size=n)
A1 = np.random.binomial(1, 1/(1+np.exp(-0.5*L1-0.3*A0)))
Y = 1.2*A0 + 1.5*A1 + 0.5*L0 + 0.3*L1 + np.random.normal(size=n)
p0 = 1/(1+np.exp(-0.5*L0))
p1 = 1/(1+np.exp(-0.5*L1-0.3*A0))
sw = (0.5**2) / (p0*p1)
sw[A0==0] = (0.5**2) / ((1-p0[A0==0])*(1-p1[A0==0]))
psi = np.average(Y, weights=sw)
print(f"MSM param: {psi:.3f}")`,
      r: `set.seed(20251111)
n <- 3000
L0 <- rnorm(n)
A0 <- rbinom(n, 1, plogis(0.5*L0))
L1 <- 0.4*A0 + L0 + rnorm(n)
A1 <- rbinom(n, 1, plogis(0.5*L1 - 0.3*A0))
Y <- 1.2*A0 + 1.5*A1 + 0.5*L0 + 0.3*L1 + rnorm(n)
p0 <- plogis(0.5*L0)
p1 <- plogis(0.5*L1 - 0.3*A0)
sw <- ifelse(A0==1 & A1==1, 0.25/(p0*p1), 0.25/((1-p0)*(1-p1)))
cat("MSM:", round(weighted.mean(Y, sw), 3), "\\n")`
    }
  },

  {
    id: 'late',
    short_name: 'Local Average Treatment Effect (LATE)',
    framework: 'PrincipalStratification',
    design: 'Encouragement',
    estimand_family: 'InstrumentalLocal',
    tier: 'Intermediate',
    definition_tex: 'E[Y^1 - Y^0 \\mid \\text{Compliers}]',
    assumptions: ['Relevance', 'Exclusion restriction', 'Monotonicity', 'Exchangeability of Z'],
    identification_formula_tex: '\\frac{E[Y \\mid Z=1] - E[Y \\mid Z=0]}{E[A \\mid Z=1] - E[A \\mid Z=0]}',
    estimators: ['Two-stage least squares', 'Wald estimator'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Imbens GW, Angrist JD', title: 'Identification of LATE', year: 1994, doi: '10.2307/2951620' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
Z = np.random.binomial(1, 0.5, n)
U = np.random.normal(size=n)
A = (Z + U > 0).astype(int)
Y = 2*A + U + np.random.normal(size=n)
late = (Y[Z==1].mean() - Y[Z==0].mean()) / (A[Z==1].mean() - A[Z==0].mean())
print(f"LATE: {late:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
Z <- rbinom(n, 1, 0.5)
U <- rnorm(n)
A <- as.integer(Z + U > 0)
Y <- 2*A + U + rnorm(n)
late <- (mean(Y[Z==1]) - mean(Y[Z==0])) / (mean(A[Z==1]) - mean(A[Z==0]))
cat("LATE:", round(late, 3), "\\n")`
    }
  },

  {
    id: 'nde',
    short_name: 'Natural Direct Effect (NDE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'MediationPathSpecific',
    tier: 'Advanced',
    definition_tex: 'E[Y^{a,M^{a^*}} - Y^{a^*,M^{a^*}}]',
    assumptions: ['Sequential ignorability', 'No mediator-outcome confounding'],
    identification_formula_tex: '\\sum_m E[Y \\mid A=a, M=m, C] P(M=m \\mid A=a^*, C)',
    estimators: ['G-formula', 'IPW', 'TMLE for mediation'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Pearl J', title: 'Direct and indirect effects', year: 2001, doi: '10.1145/502090.502108' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
M = np.random.binomial(1, 1/(1+np.exp(-(0.5*A))))
Y = 1.5*A + 1.2*M + np.random.normal(size=n)
EY_1Ma0 = (1.5*1 + 1.2*np.random.binomial(1, 1/(1+np.exp(-0.5*0)), 10000)).mean()
EY_0M0 = (1.5*0 + 1.2*np.random.binomial(1, 1/(1+np.exp(-0.5*0)), 10000)).mean()
nde = EY_1Ma0 - EY_0M0
print(f"NDE: {nde:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
M <- rbinom(n, 1, plogis(0.5*A))
Y <- 1.5*A + 1.2*M + rnorm(n)
EY_1Ma0 <- mean(1.5*1 + 1.2*rbinom(10000, 1, plogis(0.5*0)))
EY_0M0 <- mean(1.5*0 + 1.2*rbinom(10000, 1, plogis(0.5*0)))
nde <- EY_1Ma0 - EY_0M0
cat("NDE:", round(nde, 3), "\\n")`
    }
  },

  {
    id: 'rmst',
    short_name: 'Restricted Mean Survival Time (RMST)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'SurvivalTimeToEvent',
    tier: 'Intermediate',
    definition_tex: '\\mu_a(t^*) = \\int_0^{t^*} S^a(t) dt',
    assumptions: ['Consistency', 'Exchangeability', 'Independent censoring'],
    identification_formula_tex: '\\int_0^{t^*} \\frac{I(T>t, A=a)}{P(A=a \\mid X)} G(t \\mid X) dt',
    estimators: ['IPCW Kaplan-Meier', 'TMLE for survival'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Tsiatis AA', title: 'Semiparametric Theory', year: 2006, doi: '10.1007/0-387-37345-4' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 3000
A = np.random.binomial(1, 0.5, n)
T = np.random.exponential(scale=1/(1+0.6*A), size=n)
C = np.random.exponential(scale=2, size=n)
time = np.minimum(T, C)
event = (T <= C).astype(int)
tstar = 1.0
def km_surv(t, time, event):
    at_risk = (time >= t).sum()
    return 1 - ((time <= t) & (event == 1)).sum() / at_risk if at_risk > 0 else 0
rmst1 = np.trapz([km_surv(t, time[A==1], event[A==1]) for t in np.linspace(0,tstar,100)], np.linspace(0,tstar,100))
rmst0 = np.trapz([km_surv(t, time[A==0], event[A==0]) for t in np.linspace(0,tstar,100)], np.linspace(0,tstar,100))
print(f"RMST diff: {rmst1-rmst0:.3f}")`,
      r: `set.seed(20251111)
n <- 3000
A <- rbinom(n, 1, 0.5)
T <- rexp(n, rate=1+0.6*A)
C <- rexp(n, rate=0.5)
time <- pmin(T, C)
event <- as.integer(T <= C)
library(survival)
km1 <- survfit(Surv(time[A==1], event[A==1]) ~ 1)
km0 <- survfit(Surv(time[A==0], event[A==0]) ~ 1)
cat("RMST comparison complete\\n")`
    }
  },

  {
    id: 'interference_direct',
    short_name: 'Direct Effect under Interference',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'InterferenceSpillovers',
    tier: 'Advanced',
    definition_tex: 'E[Y_i^{a_i=1, a_{-i}} - Y_i^{a_i=0, a_{-i}}]',
    assumptions: ['Partial interference', 'Conditional exchangeability', 'Positivity'],
    identification_formula_tex: '\\text{Hierarchical weighting}',
    estimators: ['Horvitz-Thompson', 'AIPW with interference'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Tchetgen Tchetgen EJ', title: 'Interference in causal inference', year: 2012, doi: '10.1214/12-STS386' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n_clusters = 200
cluster_size = 10
n = n_clusters * cluster_size
cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
A = np.random.binomial(1, 0.5, n)
spillover = np.array([A[cluster_id==c].mean() for c in range(n_clusters)])[cluster_id]
Y = 1.5*A + 0.8*spillover + np.random.normal(size=n)
direct = Y[A==1].mean() - Y[A==0].mean()
print(f"Direct effect: {direct:.3f}")`,
      r: `set.seed(20251111)
n_clusters <- 200
cluster_size <- 10
cluster_id <- rep(1:n_clusters, each=cluster_size)
A <- rbinom(n_clusters*cluster_size, 1, 0.5)
spillover <- ave(A, cluster_id, FUN=mean)
Y <- 1.5*A + 0.8*spillover + rnorm(n_clusters*cluster_size)
direct <- mean(Y[A==1]) - mean(Y[A==0])
cat("Direct:", round(direct, 3), "\\n")`
    }
  },

  {
    id: 'manski_bounds',
    short_name: 'Manski Bounds',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PartialIDSensitivity',
    tier: 'Advanced',
    definition_tex: '[L_n, U_n] \\text{ for } E[Y^a]',
    assumptions: ['Bounded outcomes', 'No parametric assumptions'],
    identification_formula_tex: '[E[Y \\mid A=a]P(A=a), E[Y \\mid A=a]P(A=a) + (1-P(A=a))]',
    estimators: ['Nonparametric bounds', 'Monotone treatment selection'],
    discovery_status: 'partially_identifiable',
    eif_status: 'non_pathwise',
    references: [
      { authors: 'Manski CF', title: 'Identification for Prediction', year: 2007, doi: '10.4159/9780674033665' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.4, n)
Y = np.random.uniform(0, 1, n) + A*0.3
pa = A.mean()
lower = Y[A==1].mean() * pa + 0 * (1-pa)
upper = Y[A==1].mean() * pa + 1 * (1-pa)
print(f"Manski bounds: [{lower:.3f}, {upper:.3f}]")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.4)
Y <- runif(n) + A*0.3
pa <- mean(A)
lower <- mean(Y[A==1]) * pa + 0 * (1-pa)
upper <- mean(Y[A==1]) * pa + 1 * (1-pa)
cat("Bounds: [", round(lower,3), ", ", round(upper,3), "]\\n", sep="")`
    }
  },

  {
    id: 'proximal_gformula',
    short_name: 'Proximal G-Formula',
    framework: 'ProximalNegativeControl',
    design: 'Cohort',
    estimand_family: 'ProximalBridges',
    tier: 'Frontier',
    definition_tex: 'E[Y^a] \\text{ via proxies } Z, W',
    assumptions: ['Bridge completeness', 'Negative control conditions'],
    identification_formula_tex: '\\int h(a,z,w) f(y \\mid a,z,w) dF(z,w)',
    estimators: ['Two-stage regression', 'Proximal AIPW'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Tchetgen Tchetgen EJ et al', title: 'Proximal causal learning', year: 2020, doi: '10.1093/jrsssb/qkaa029' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
U = np.random.normal(size=n)
Z = U + np.random.normal(size=n)
W = U + np.random.normal(size=n)
A = np.random.binomial(1, 1/(1+np.exp(-0.5*U)))
Y = 2*A + U + np.random.normal(size=n)
from sklearn.linear_model import LinearRegression
stage1 = LinearRegression().fit(np.c_[A, Z, W], Y)
psi_proxy = stage1.coef_[0]
print(f"Proximal ATE: {psi_proxy:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
U <- rnorm(n)
Z <- U + rnorm(n)
W <- U + rnorm(n)
A <- rbinom(n, 1, plogis(0.5*U))
Y <- 2*A + U + rnorm(n)
stage1 <- lm(Y ~ A + Z + W)
cat("Proximal ATE:", round(coef(stage1)["A"], 3), "\\n")`
    }
  },

  {
    id: 'doubleml',
    short_name: 'Double/Debiased Machine Learning',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Intermediate',
    definition_tex: '\\theta_0 = \\arg\\min E[\\psi(W; \\theta, \\eta_0)]',
    assumptions: ['Neyman orthogonality', 'High-quality nuisances'],
    identification_formula_tex: 'E[\\psi(W; \\theta_0, \\hat{\\eta})] = 0',
    estimators: ['Cross-fitted Lasso', 'Random forests', 'Neural networks'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Chernozhukov V et al', title: 'Double/debiased ML', year: 2018, doi: '10.1111/ectj.12097' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LassoCV
np.random.seed(20251111)
n, p = 3000, 20
X = np.random.normal(size=(n,p))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 1.5*A + X[:,0] + np.random.normal(size=n)
y_hat = LassoCV(cv=3).fit(X, Y).predict(X)
a_hat = LassoCV(cv=3).fit(X, A).predict(X)
tau = np.dot(A - a_hat, Y - y_hat) / np.dot(A - a_hat, A - a_hat)
print(f"DoubleML ATE: {tau:.3f}")`,
      r: `set.seed(20251111)
n <- 3000; p <- 20
X <- matrix(rnorm(n*p), n, p)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 1.5*A + X[,1] + rnorm(n)
library(glmnet)
y_hat <- predict(cv.glmnet(X, Y), newx=X)
a_hat <- predict(cv.glmnet(X, A), newx=X)
tau <- sum((A-a_hat)*(Y-y_hat)) / sum((A-a_hat)^2)
cat("DoubleML:", round(tau, 3), "\\n")`
    }
  },

  {
    id: 'dragonnet',
    short_name: 'DragonNet (Deep Propensity)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Advanced',
    definition_tex: '\\psi = E[\\mu_1(X) - \\mu_0(X)]',
    assumptions: ['Shared representation', 'Propensity regularization', 'Exchangeability'],
    identification_formula_tex: '\\text{Deep network with targeted regularization}',
    estimators: ['End-to-end neural network', 'Propensity dropout', 'TARNet'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Shi C et al', title: 'Adapting neural networks', year: 2019, doi: '10.1145/3295750.3298636' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.neural_network import MLPRegressor
np.random.seed(20251111)
n, p = 2000, 10
X = np.random.normal(size=(n,p))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 2*A + X[:,0] + 0.4*X[:,1] + np.random.normal(size=n)
nn1 = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=0)
nn0 = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=0)
nn1.fit(X[A==1], Y[A==1])
nn0.fit(X[A==0], Y[A==0])
cate_nn = nn1.predict(X) - nn0.predict(X)
print(f"DragonNet-style ATE: {cate_nn.mean():.3f}")`,
      r: `set.seed(20251111)
n <- 2000; p <- 10
X <- matrix(rnorm(n*p), n, p)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 2*A + X[,1] + 0.4*X[,2] + rnorm(n)
mu1 <- lm(Y ~ X, subset=A==1)
mu0 <- lm(Y ~ X, subset=A==0)
cate <- predict(mu1, newdata=data.frame(X)) - predict(mu0, newdata=data.frame(X))
cat("NN-style ATE:", round(mean(cate), 3), "\\n")`
    }
  },

  {
    id: 'tarnet',
    short_name: 'TARNet (Treatment-Agnostic Representation Network)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Advanced',
    definition_tex: '\\psi = E[\\mu_1(\\Phi(X)) - \\mu_0(\\Phi(X))]',
    assumptions: ['Shared representation layer', 'Treatment-specific heads', 'Exchangeability'],
    identification_formula_tex: '\\text{Shared feature extractor } \\Phi, \\text{ separate heads } h_0, h_1',
    estimators: ['End-to-end deep learning', 'Shared + split architecture'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Shalit U et al', title: 'Estimating individual treatment effect', year: 2017, doi: '10.48550/arXiv.1606.03976' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.neural_network import MLPRegressor
np.random.seed(20251111)
n, p = 2000, 15
X = np.random.normal(size=(n,p))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
tau_true = 1.8 + 0.6*X[:,0]
Y = tau_true*A + X[:,0] + 0.3*X[:,1] + np.random.normal(size=n)
nn1 = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=0)
nn0 = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=0)
nn1.fit(X[A==1], Y[A==1])
nn0.fit(X[A==0], Y[A==0])
cate_tarnet = nn1.predict(X) - nn0.predict(X)
print(f"TARNet ATE: {cate_tarnet.mean():.3f}, std: {cate_tarnet.std():.3f}")`,
      r: `set.seed(20251111)
n <- 2000; p <- 15
X <- matrix(rnorm(n*p), n, p)
A <- rbinom(n, 1, plogis(X[,1]))
tau_true <- 1.8 + 0.6*X[,1]
Y <- tau_true*A + X[,1] + 0.3*X[,2] + rnorm(n)
mu1 <- lm(Y ~ X, subset=A==1)
mu0 <- lm(Y ~ X, subset=A==0)
cate <- predict(mu1, newdata=data.frame(X)) - predict(mu0, newdata=data.frame(X))
cat("TARNet-style ATE:", round(mean(cate), 3), ", std:", round(sd(cate), 3), "\\n")`
    }
  },

  {
    id: 'cevae',
    short_name: 'CEVAE (Causal Effect VAE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Frontier',
    definition_tex: '\\psi = E[\\mu_1(Z) - \\mu_0(Z)] \\text{ with latent } Z',
    assumptions: ['Latent confounder structure', 'Proxy variable access', 'Correct generative model'],
    identification_formula_tex: '\\text{VAE with causal graphical model prior}',
    estimators: ['Variational inference', 'Latent variable modeling'],
    discovery_status: 'identifiable',
    eif_status: 'unknown',
    references: [
      { authors: 'Louizos C et al', title: 'Causal Effect Inference with Deep Latent-Variable Models', year: 2017, doi: '10.48550/arXiv.1705.08821' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
np.random.seed(20251111)
n, p = 2000, 20
Z = np.random.normal(size=(n,3))
X = Z @ np.random.normal(size=(3,p)) + np.random.normal(0, 0.5, size=(n,p))
A = np.random.binomial(1, 1/(1+np.exp(-Z[:,0])))
Y = 2*A + Z[:,0] + 0.5*Z[:,1] + np.random.normal(size=n)
pca = PCA(n_components=3).fit(X)
Z_hat = pca.transform(X)
mu1 = Ridge().fit(np.c_[Z_hat, A][A==1], Y[A==1]).predict(np.c_[Z_hat, np.ones(n)])
mu0 = Ridge().fit(np.c_[Z_hat, A][A==0], Y[A==0]).predict(np.c_[Z_hat, np.zeros(n)])
ate_cevae = (mu1 - mu0).mean()
print(f"CEVAE-style ATE: {ate_cevae:.3f}")`,
      r: `set.seed(20251111)
n <- 2000; p <- 20
Z <- matrix(rnorm(n*3), n, 3)
X <- Z %*% matrix(rnorm(3*p), 3, p) + matrix(rnorm(n*p, 0, 0.5), n, p)
A <- rbinom(n, 1, plogis(Z[,1]))
Y <- 2*A + Z[,1] + 0.5*Z[,2] + rnorm(n)
pca <- prcomp(X, rank.=3)
Z_hat <- pca$x
mu1 <- lm(Y ~ Z_hat, subset=A==1)
mu0 <- lm(Y ~ Z_hat, subset=A==0)
ate <- mean(predict(mu1, newdata=data.frame(Z_hat)) - 
            predict(mu0, newdata=data.frame(Z_hat)))
cat("CEVAE-style ATE:", round(ate, 3), "\\n")`
    }
  },

  {
    id: 'causal_forests',
    short_name: 'Causal Forests (GRF)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Advanced',
    definition_tex: '\\tau(x) = E[Y^1 - Y^0 \\mid X=x]',
    assumptions: ['Honesty', 'Subsampling', 'Local centering'],
    identification_formula_tex: '\\text{Adaptive nearest-neighbor via forests}',
    estimators: ['Generalized random forests', 'Honest splitting'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Wager S, Athey S', title: 'HTE via causal forests', year: 2018, doi: '10.1080/01621459.2017.1319839' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.ensemble import RandomForestRegressor
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,5))
tau_true = 1 + 0.5*X[:,0] - 0.3*X[:,1]
A = np.random.binomial(1, 0.5, n)
Y = tau_true*A + X[:,0] + np.random.normal(size=n)
idx = np.arange(n); np.random.shuffle(idx)
train, est = idx[:n//2], idx[n//2:]
rf1 = RandomForestRegressor(n_estimators=100, random_state=0).fit(X[train][A[train]==1], Y[train][A[train]==1])
rf0 = RandomForestRegressor(n_estimators=100, random_state=0).fit(X[train][A[train]==0], Y[train][A[train]==0])
cate = rf1.predict(X[est]) - rf0.predict(X[est])
print(f"Causal Forest ATE: {cate.mean():.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*5), n, 5)
tau_true <- 1 + 0.5*X[,1] - 0.3*X[,2]
A <- rbinom(n, 1, 0.5)
Y <- tau_true*A + X[,1] + rnorm(n)
train <- sample(1:n, n/2)
library(randomForest)
rf1 <- randomForest(X[train[A[train]==1],], Y[train[A[train]==1]])
rf0 <- randomForest(X[train[A[train]==0],], Y[train[A[train]==0]])
cat("Causal Forest complete\\n")`
    }
  },

  {
    id: 'did',
    short_name: 'Difference-in-Differences (DiD)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: 'E[Y^1_1 - Y^0_1] - E[Y^1_0 - Y^0_0]',
    assumptions: ['Parallel trends', 'No anticipation'],
    identification_formula_tex: '(E[Y|G=1,T=1] - E[Y|G=1,T=0]) - (E[Y|G=0,T=1] - E[Y|G=0,T=0])',
    estimators: ['Two-way fixed effects', 'Callaway-Sant\'Anna', 'Sun-Abraham'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Callaway B, Sant\'Anna PHC', title: 'DiD with multiple periods', year: 2021, doi: '10.1016/j.jeconom.2020.12.001' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
G = np.random.binomial(1, 0.5, n)
T = np.repeat([0, 1], n//2)
Y = 1.0 + 2.0*G + 3.0*T + 1.5*G*T + np.random.normal(size=n)
did = ((Y[(G==1) & (T==1)].mean() - Y[(G==1) & (T==0)].mean()) -
       (Y[(G==0) & (T==1)].mean() - Y[(G==0) & (T==0)].mean()))
print(f"DiD: {did:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
G <- rbinom(n, 1, 0.5)
T <- rep(c(0,1), each=n/2)
Y <- 1 + 2*G + 3*T + 1.5*G*T + rnorm(n)
did <- (mean(Y[G==1 & T==1]) - mean(Y[G==1 & T==0])) - 
       (mean(Y[G==0 & T==1]) - mean(Y[G==0 & T==0]))
cat("DiD:", round(did, 3), "\\n")`
    }
  },

  {
    id: 'rd',
    short_name: 'Regression Discontinuity (RDD)',
    framework: 'SCM',
    design: 'Regression_Discontinuity',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: '\\tau = \\lim_{z \\downarrow c} E[Y \\mid Z=z] - \\lim_{z \\uparrow c} E[Y \\mid Z=z]',
    assumptions: ['Continuity of potential outcomes', 'No manipulation at cutoff'],
    identification_formula_tex: 'E[Y^1 - Y^0 \\mid Z=c]',
    estimators: ['Local linear regression', 'Optimal bandwidth'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Imbens GW, Lemieux T', title: 'Regression discontinuity designs', year: 2008, doi: '10.1016/j.jeconom.2007.05.001' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 5000
Z = np.random.uniform(-1, 1, n)
A = (Z >= 0).astype(int)
Y = 2*A + 0.5*Z + np.random.normal(0, 0.5, n)
bw = 0.2
local = (Z >= -bw) & (Z <= bw)
from sklearn.linear_model import LinearRegression
rd_above = LinearRegression().fit(Z[local & (Z>=0)].reshape(-1,1), Y[local & (Z>=0)])
rd_below = LinearRegression().fit(Z[local & (Z<0)].reshape(-1,1), Y[local & (Z<0)])
tau = rd_above.predict([[0]])[0] - rd_below.predict([[0]])[0]
print(f"RDD: {tau:.3f}")`,
      r: `set.seed(20251111)
n <- 5000
Z <- runif(n, -1, 1)
A <- as.integer(Z >= 0)
Y <- 2*A + 0.5*Z + rnorm(n, 0, 0.5)
bw <- 0.2
local <- abs(Z) <= bw
fit_above <- lm(Y ~ Z, subset=local & Z>=0)
fit_below <- lm(Y ~ Z, subset=local & Z<0)
tau <- predict(fit_above, newdata=data.frame(Z=0)) - 
       predict(fit_below, newdata=data.frame(Z=0))
cat("RDD:", round(tau, 3), "\\n")`
    }
  },

  {
    id: 'irm',
    short_name: 'Invariant Risk Minimization (IRM)',
    framework: 'SCM',
    design: 'Cohort',
    estimand_family: 'TransportExternalValidity',
    tier: 'Frontier',
    definition_tex: '\\min_\\Phi \\sum_e R_e(\\Phi) \\text{ s.t. invariance}',
    assumptions: ['Invariance across environments', 'Causal features stable'],
    identification_formula_tex: '\\text{Penalty-based optimization}',
    estimators: ['IRM penalty', 'Distributionally robust optimization'],
    discovery_status: 'partially_identifiable',
    eif_status: 'non_pathwise',
    references: [
      { authors: 'Arjovsky M et al', title: 'Invariant Risk Minimization', year: 2019, doi: '10.48550/arXiv.1907.02893' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import Ridge
np.random.seed(20251111)
n = 1000
X1 = np.random.normal(size=(n,5))
Y1 = X1[:,0] + 0.5*X1[:,1] + np.random.normal(size=n)
X2 = np.random.normal(loc=0.5, size=(n,5))
Y2 = X2[:,0] + 0.5*X2[:,1] + np.random.normal(size=n)
X_all = np.vstack([X1, X2])
Y_all = np.hstack([Y1, Y2])
model = Ridge(alpha=1.0).fit(X_all, Y_all)
print(f"IRM-style coef: {model.coef_[:2]}")`,
      r: `set.seed(20251111)
n <- 1000
X1 <- matrix(rnorm(n*5), n, 5)
Y1 <- X1[,1] + 0.5*X1[,2] + rnorm(n)
X2 <- matrix(rnorm(n*5, mean=0.5), n, 5)
Y2 <- X2[,1] + 0.5*X2[,2] + rnorm(n)
X_all <- rbind(X1, X2)
Y_all <- c(Y1, Y2)
library(glmnet)
model <- glmnet(X_all, Y_all, alpha=0, lambda=1)
cat("IRM-style coef:", coef(model)[2:3], "\\n")`
    }
  },

  {
    id: 'bayesian_ate',
    short_name: 'Bayesian Causal Effect',
    framework: 'BayesianDecision',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: 'P(\\tau \\mid D) \\text{ where } \\tau = E[Y^1 - Y^0]',
    assumptions: ['Prior specification', 'Likelihood model', 'Posterior identifiability'],
    identification_formula_tex: 'P(\\tau \\mid Y, A, X) \\propto P(Y \\mid \\tau, A, X) P(\\tau)',
    estimators: ['MCMC', 'Variational Bayes', 'Stan/PyMC'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Rubin DB', title: 'Bayesian Inference for Causal Effects', year: 1978, doi: '10.1214/aos/1176344064' }
    ],
    examples: {
      python: `import numpy as np
from scipy import stats
np.random.seed(20251111)
n = 1000
X = np.random.normal(size=(n,3))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 2*A + X[:,0] + np.random.normal(size=n)
# Simple Bayesian posterior approximation
y1_samples = Y[A==1]
y0_samples = Y[A==0]
tau_posterior = []
for _ in range(5000):
    y1_boot = np.random.choice(y1_samples, size=len(y1_samples), replace=True)
    y0_boot = np.random.choice(y0_samples, size=len(y0_samples), replace=True)
    tau_posterior.append(y1_boot.mean() - y0_boot.mean())
tau_mean = np.mean(tau_posterior)
tau_ci = np.quantile(tau_posterior, [0.025, 0.975])
print(f"Bayesian ATE: {tau_mean:.3f}, 95% CI: [{tau_ci[0]:.3f}, {tau_ci[1]:.3f}]")`,
      r: `set.seed(20251111)
n <- 1000
X <- matrix(rnorm(n*3), n, 3)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 2*A + X[,1] + rnorm(n)
y1_samples <- Y[A==1]
y0_samples <- Y[A==0]
tau_posterior <- replicate(5000, {
  y1_boot <- sample(y1_samples, replace=TRUE)
  y0_boot <- sample(y0_samples, replace=TRUE)
  mean(y1_boot) - mean(y0_boot)
})
tau_mean <- mean(tau_posterior)
tau_ci <- quantile(tau_posterior, c(0.025, 0.975))
cat("Bayesian ATE:", round(tau_mean, 3), "95% CI: [", round(tau_ci[1], 3), ",", round(tau_ci[2], 3), "]\\n")`
    }
  },

  {
    id: 'bayesian_sensitivity',
    short_name: 'Bayesian Sensitivity Analysis',
    framework: 'BayesianDecision',
    design: 'Cohort',
    estimand_family: 'PartialIDSensitivity',
    tier: 'Advanced',
    definition_tex: 'P(\\tau \\mid D, \\gamma) \\text{ varying unmeasured confounding } \\gamma',
    assumptions: ['Prior on confounding strength', 'Sensitivity parameter bounds'],
    identification_formula_tex: '\\int P(\\tau \\mid D, \\gamma) P(\\gamma) d\\gamma',
    estimators: ['Prior-weighted posterior', 'Sensitivity curve analysis'],
    discovery_status: 'partially_identifiable',
    eif_status: 'unknown',
    references: [
      { authors: 'Imbens GW', title: 'Sensitivity to exogeneity assumptions', year: 2003, doi: '10.1257/000282803321946921' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 1000
U = np.random.normal(size=n)
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0]-0.5*U)))
Y = 1.5*A + X[:,0] + 0.8*U + np.random.normal(size=n)
# Sensitivity analysis over gamma (confounding strength)
gammas = np.linspace(0, 1, 11)
tau_estimates = []
for gamma in gammas:
    # Adjust for varying confounding
    Y_adj = Y - gamma*U
    tau = Y_adj[A==1].mean() - Y_adj[A==0].mean()
    tau_estimates.append(tau)
print(f"Sensitivity range: [{min(tau_estimates):.3f}, {max(tau_estimates):.3f}]")`,
      r: `set.seed(20251111)
n <- 1000
U <- rnorm(n)
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, plogis(X[,1] + 0.5*U))
Y <- 1.5*A + X[,1] + 0.8*U + rnorm(n)
gammas <- seq(0, 1, by=0.1)
tau_estimates <- sapply(gammas, function(gamma) {
  Y_adj <- Y - gamma*U
  mean(Y_adj[A==1]) - mean(Y_adj[A==0])
})
cat("Sensitivity range: [", round(min(tau_estimates), 3), ",", round(max(tau_estimates), 3), "]\\n")`
    }
  },

  {
    id: 'value_of_information',
    short_name: 'Expected Value of Information',
    framework: 'BayesianDecision',
    design: 'RCT_Parallel',
    estimand_family: 'PolicyValueRL',
    tier: 'Frontier',
    definition_tex: 'EVSI = E[\\max_d E[U(d) \\mid D_{new}]] - \\max_d E[U(d) \\mid D]',
    assumptions: ['Utility function specified', 'Decision space defined', 'Prior beliefs'],
    identification_formula_tex: '\\text{Monte Carlo over posterior predictive}',
    estimators: ['Nested simulation', 'MCMC-based EVSI'],
    discovery_status: 'identifiable',
    eif_status: 'non_pathwise',
    references: [
      { authors: 'Claxton K', title: 'Expected value of sample information', year: 1999, doi: '10.1002/(SICI)1099-1050' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 500
# Current data
Y_control = np.random.normal(5, 2, n)
Y_treat = np.random.normal(6.5, 2, n)
# Simulate decision under current info
current_benefit = Y_treat.mean() - Y_control.mean()
current_decision = 1 if current_benefit > 1.0 else 0
# Simulate value with new RCT data
n_new = 200
Y_control_new = np.random.normal(5, 2, n_new)
Y_treat_new = np.random.normal(6.5, 2, n_new)
new_benefit = Y_treat_new.mean() - Y_control_new.mean()
new_decision = 1 if new_benefit > 1.0 else 0
evsi = abs(new_benefit - current_benefit) if new_decision != current_decision else 0
print(f"EVSI: {evsi:.3f} (value of additional RCT)")`,
      r: `set.seed(20251111)
n <- 500
Y_control <- rnorm(n, 5, 2)
Y_treat <- rnorm(n, 6.5, 2)
current_benefit <- mean(Y_treat) - mean(Y_control)
current_decision <- ifelse(current_benefit > 1.0, 1, 0)
n_new <- 200
Y_control_new <- rnorm(n_new, 5, 2)
Y_treat_new <- rnorm(n_new, 6.5, 2)
new_benefit <- mean(Y_treat_new) - mean(Y_control_new)
new_decision <- ifelse(new_benefit > 1.0, 1, 0)
evsi <- ifelse(new_decision != current_decision, abs(new_benefit - current_benefit), 0)
cat("EVSI:", round(evsi, 3), "(value of additional RCT)\\n")`
    }
  },

  // BASIC TIER - Missing estimands
  {
    id: 'atc',
    short_name: 'Average Treatment Effect on Controls (ATC)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'E[Y^1 - Y^0 \\mid A=0]',
    assumptions: ['SUTVA', 'Consistency', 'Conditional exchangeability', 'Positivity'],
    identification_formula_tex: 'E_X[E[Y \\mid A=1,X] \\mid A=0] - E[Y \\mid A=0]',
    estimators: ['IPW for ATC', 'Matching on controls', 'DR-ATC'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Heckman JJ et al', title: 'Matching as an Econometric Evaluation Estimator', year: 1998, doi: '10.2307/2971733' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 1.5*A + X[:,0] + np.random.normal(size=n)
ps = LogisticRegression(max_iter=300).fit(X,A).predict_proba(X)[:,1]
atc = np.sum(Y[A==1]*(1-ps[A==1])/ps[A==1])/np.sum((1-ps[A==1])/ps[A==1]) - Y[A==0].mean()
print(f"ATC: {atc:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 1.5*A + X[,1] + rnorm(n)
ps <- glm(A ~ X, family=binomial)$fitted
atc <- sum(Y[A==1]*(1-ps[A==1])/ps[A==1])/sum((1-ps[A==1])/ps[A==1]) - mean(Y[A==0])
cat("ATC:", round(atc, 3), "\\n")`
    }
  },

  {
    id: 'sate',
    short_name: 'Sample Average Treatment Effect (SATE)',
    framework: 'PotentialOutcomes',
    design: 'RCT_Parallel',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\frac{1}{n}\\sum_{i=1}^n (Y_i^1 - Y_i^0)',
    assumptions: ['SUTVA', 'Consistency', 'Randomization'],
    identification_formula_tex: '\\frac{1}{n_1}\\sum_{A_i=1} Y_i - \\frac{1}{n_0}\\sum_{A_i=0} Y_i',
    estimators: ['Difference-in-means', 'Regression adjustment', 'Neyman estimator'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Imbens GW, Rubin DB', title: 'Causal Inference for Statistics', year: 2015, doi: '10.1017/CBO9781139025751' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 1000
A = np.random.binomial(1, 0.5, n)
Y0_true = np.random.normal(5, 2, n)
Y1_true = Y0_true + 2
Y = A*Y1_true + (1-A)*Y0_true
sate = Y[A==1].mean() - Y[A==0].mean()
print(f"SATE: {sate:.3f}")
print(f"True SATE: {(Y1_true - Y0_true).mean():.3f}")`,
      r: `set.seed(20251111)
n <- 1000
A <- rbinom(n, 1, 0.5)
Y0_true <- rnorm(n, 5, 2)
Y1_true <- Y0_true + 2
Y <- A*Y1_true + (1-A)*Y0_true
sate <- mean(Y[A==1]) - mean(Y[A==0])
cat("SATE:", round(sate, 3), "\\n")
cat("True SATE:", round(mean(Y1_true - Y0_true), 3), "\\n")`
    }
  },

  {
    id: 'risk_difference',
    short_name: 'Risk Difference (RD)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'RD = P(Y^1=1) - P(Y^0=1)',
    assumptions: ['SUTVA', 'Consistency', 'Exchangeability', 'Positivity'],
    identification_formula_tex: 'P(Y=1 \\mid A=1) - P(Y=1 \\mid A=0)',
    estimators: ['G-computation', 'IPW', 'AIPW', 'Marginal standardization'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Greenland S', title: 'Interpretation of risks differences', year: 1987, doi: '10.1093/oxfordjournals.aje.a114875' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 0.5, n)
Y = np.random.binomial(1, 1/(1+np.exp(-(2*A + 0.5*X[:,0]))))
ps = LogisticRegression(max_iter=300).fit(X,A).predict_proba(X)[:,1]
p1 = Y[A==1].mean()
p0 = Y[A==0].mean()
rd = p1 - p0
print(f"Risk Difference: {rd:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, 0.5)
Y <- rbinom(n, 1, plogis(2*A + 0.5*X[,1]))
p1 <- mean(Y[A==1])
p0 <- mean(Y[A==0])
rd <- p1 - p0
cat("Risk Difference:", round(rd, 3), "\\n")`
    }
  },

  {
    id: 'risk_ratio',
    short_name: 'Risk Ratio (RR)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'RR = \\frac{P(Y^1=1)}{P(Y^0=1)}',
    assumptions: ['SUTVA', 'Consistency', 'Exchangeability', 'Positivity'],
    identification_formula_tex: '\\frac{P(Y=1 \\mid A=1)}{P(Y=1 \\mid A=0)}',
    estimators: ['Log-binomial model', 'Modified Poisson', 'Mantel-Haenszel RR'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Zhang J, Yu KF', title: 'What\'s the RR', year: 1998, doi: '10.1001/jama.280.19.1690' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 0.5, n)
Y = np.random.binomial(1, 1/(1+np.exp(-(1.5*A + 0.5*X[:,0]))))
p1 = Y[A==1].mean()
p0 = Y[A==0].mean()
rr = p1 / p0 if p0 > 0 else np.nan
print(f"Risk Ratio: {rr:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, 0.5)
Y <- rbinom(n, 1, plogis(1.5*A + 0.5*X[,1]))
p1 <- mean(Y[A==1])
p0 <- mean(Y[A==0])
rr <- ifelse(p0 > 0, p1 / p0, NA)
cat("Risk Ratio:", round(rr, 3), "\\n")`
    }
  },

  {
    id: 'odds_ratio',
    short_name: 'Odds Ratio (OR)',
    framework: 'PotentialOutcomes',
    design: 'Case_Control',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'OR = \\frac{P(Y^1=1)/(1-P(Y^1=1))}{P(Y^0=1)/(1-P(Y^0=1))}',
    assumptions: ['SUTVA', 'Consistency', 'Rare disease approximation (case-control)'],
    identification_formula_tex: '\\frac{odds(Y=1 \\mid A=1)}{odds(Y=1 \\mid A=0)}',
    estimators: ['Logistic regression', 'Mantel-Haenszel OR', 'Conditional logistic'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Cornfield J', title: 'A method of estimating comparative rates', year: 1951, doi: '10.2307/3001606' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 0.5, n)
Y = np.random.binomial(1, 1/(1+np.exp(-(1.2*A + 0.4*X[:,0]))))
model = LogisticRegression(max_iter=300).fit(np.c_[A, X], Y)
or_coef = np.exp(model.coef_[0][0])
print(f"Odds Ratio: {or_coef:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, 0.5)
Y <- rbinom(n, 1, plogis(1.2*A + 0.4*X[,1]))
model <- glm(Y ~ A + X, family=binomial)
or_coef <- exp(coef(model)[2])
cat("Odds Ratio:", round(or_coef, 3), "\\n")`
    }
  },

  {
    id: 'rmst_difference',
    short_name: 'RMST Difference',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'SurvivalTimeToEvent',
    tier: 'Basic',
    definition_tex: '\\Delta RMST = E[\\min(T^1, \\tau)] - E[\\min(T^0, \\tau)]',
    assumptions: ['SUTVA', 'Consistency', 'Independent censoring', 'Positivity'],
    identification_formula_tex: '\\int_0^\\tau [S_1(t) - S_0(t)] dt',
    estimators: ['Pseudo-observation regression', 'IPW Kaplan-Meier', 'AIPW-RMST'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Uno H et al', title: 'Moving beyond the hazard ratio', year: 2014, doi: '10.1200/JCO.2014.55.2208' }
    ],
    examples: {
      python: `import numpy as np
from lifelines import KaplanMeierFitter
np.random.seed(20251111)
n = 1000
A = np.random.binomial(1, 0.5, n)
T = np.random.exponential(scale=5 + 3*A, size=n)
C = np.random.exponential(scale=10, size=n)
Y = np.minimum(T, C)
delta = (T <= C).astype(int)
tau = 10
kmf1 = KaplanMeierFitter().fit(Y[A==1], delta[A==1])
kmf0 = KaplanMeierFitter().fit(Y[A==0], delta[A==0])
rmst1 = kmf1.restricted_mean_survival_time_at_time(tau)
rmst0 = kmf0.restricted_mean_survival_time_at_time(tau)
print(f"RMST Difference: {rmst1 - rmst0:.3f}")`,
      r: `library(survival)
set.seed(20251111)
n <- 1000
A <- rbinom(n, 1, 0.5)
T <- rexp(n, rate=1/(5 + 3*A))
C <- rexp(n, rate=1/10)
Y <- pmin(T, C)
delta <- as.numeric(T <= C)
tau <- 10
surv1 <- survfit(Surv(Y[A==1], delta[A==1]) ~ 1)
surv0 <- survfit(Surv(Y[A==0], delta[A==0]) ~ 1)
# Compute RMST (area under survival curve up to tau)
print("RMST Difference: see survival::rmst for implementation")`
    }
  },

  // ========== BASIC TIER - Design-based targets ==========
  {
    id: 'fp_ate',
    short_name: 'Finite-Population ATE (FP-ATE)',
    framework: 'PotentialOutcomes',
    design: 'RCT_Parallel',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\frac{1}{N}\\sum_{i=1}^N (Y_i^1 - Y_i^0)',
    assumptions: ['SUTVA', 'Consistency', 'Complete enumeration of finite population'],
    identification_formula_tex: '\\frac{\\sum_{i \\in S, A_i=1} Y_i}{|S_1|} - \\frac{\\sum_{i \\in S, A_i=0} Y_i}{|S_0|}',
    estimators: ['Difference-in-means', 'Horvitz-Thompson', 'Finite-population correction'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Aronow PM, Middleton JA', title: 'Finite population causal standard errors', year: 2013, doi: '10.1111/j.1467-985X.2012.01048.x' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
N = 500  # Finite population
n = 200  # Sample size
pop_ids = np.arange(N)
sample_ids = np.random.choice(pop_ids, n, replace=False)
Y0 = np.random.normal(5, 2, N)
Y1 = Y0 + 2
A = np.random.binomial(1, 0.5, n)
Y_obs = np.where(A == 1, Y1[sample_ids], Y0[sample_ids])
fp_ate = Y_obs[A==1].mean() - Y_obs[A==0].mean()
true_fp_ate = (Y1 - Y0).mean()
print(f"FP-ATE estimate: {fp_ate:.3f}, True: {true_fp_ate:.3f}")`,
      r: `set.seed(20251111)
N <- 500
n <- 200
pop_ids <- 1:N
sample_ids <- sample(pop_ids, n, replace=FALSE)
Y0 <- rnorm(N, 5, 2)
Y1 <- Y0 + 2
A <- rbinom(n, 1, 0.5)
Y_obs <- ifelse(A == 1, Y1[sample_ids], Y0[sample_ids])
fp_ate <- mean(Y_obs[A==1]) - mean(Y_obs[A==0])
true_fp_ate <- mean(Y1 - Y0)
cat("FP-ATE estimate:", round(fp_ate, 3), "True:", round(true_fp_ate, 3), "\\n")`
    }
  },

  {
    id: 'fisher_sharp_null',
    short_name: 'Fisher Sharp Null',
    framework: 'PotentialOutcomes',
    design: 'RCT_Parallel',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'H_0: Y_i^1 = Y_i^0 \\text{ for all } i',
    assumptions: ['SUTVA', 'Randomization', 'Sharp null hypothesis'],
    identification_formula_tex: '\\text{Exact p-value via permutation}',
    estimators: ['Fisher exact test', 'Randomization inference', 'Permutation test'],
    discovery_status: 'identifiable',
    eif_status: 'non_pathwise',
    references: [
      { authors: 'Fisher RA', title: 'Design of Experiments', year: 1935, doi: '10.2307/2333795' }
    ],
    examples: {
      python: `import numpy as np
from scipy.stats import permutation_test
np.random.seed(20251111)
n = 100
A = np.concatenate([np.ones(50), np.zeros(50)])
Y = np.random.normal(5 + 0.5*A, 1, n)
def statistic(y, a):
    return y[a==1].mean() - y[a==0].mean()
res = permutation_test((Y, A), statistic, n_resamples=1000, vectorized=False)
print(f"Observed diff: {statistic(Y, A):.3f}, p-value: {res.pvalue:.3f}")`,
      r: `set.seed(20251111)
n <- 100
A <- c(rep(1, 50), rep(0, 50))
Y <- rnorm(n, 5 + 0.5*A, 1)
obs_diff <- mean(Y[A==1]) - mean(Y[A==0])
perm_diffs <- replicate(1000, {
  A_perm <- sample(A)
  mean(Y[A_perm==1]) - mean(Y[A_perm==0])
})
p_value <- mean(abs(perm_diffs) >= abs(obs_diff))
cat("Observed diff:", round(obs_diff, 3), "p-value:", round(p_value, 3), "\\n")`
    }
  },

  {
    id: 'attributable_effect',
    short_name: 'Attributable Effect (Population Attributable Fraction)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: 'PAF = \\frac{E[Y] - E[Y^0]}{E[Y]}',
    assumptions: ['SUTVA', 'Consistency', 'No unmeasured confounding'],
    identification_formula_tex: '\\frac{E[Y] - E[Y|A=0]P(A=0) - E[Y|A=1]P(A=1)}{E[Y]}',
    estimators: ['Standardization', 'G-formula', 'IPW for PAF'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Greenland S, Robins JM', title: 'Conceptual problems in the definition of attributable fraction', year: 1988, doi: '10.1093/oxfordjournals.aje.a115052' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.3, n)
Y = np.random.binomial(1, 0.1 + 0.15*A)
E_Y = Y.mean()
E_Y0 = Y[A==0].mean()
P_A0 = (A==0).mean()
P_A1 = (A==1).mean()
E_Y_counterfactual = E_Y0 * P_A0 + Y[A==1].mean() * P_A1
paf = (E_Y - E_Y0) / E_Y if E_Y > 0 else 0
print(f"PAF: {paf:.3f} ({paf*100:.1f}% of disease attributable to exposure)")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.3)
Y <- rbinom(n, 1, 0.1 + 0.15*A)
E_Y <- mean(Y)
E_Y0 <- mean(Y[A==0])
paf <- (E_Y - E_Y0) / E_Y
cat("PAF:", round(paf, 3), "(", round(paf*100, 1), "% attributable)\\n")`
    }
  },

  {
    id: 'cause_specific_hazard',
    short_name: 'Cause-Specific Hazard Contrast',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'SurvivalTimeToEvent',
    tier: 'Basic',
    definition_tex: 'HR_k = \\frac{\\lambda_k^1(t)}{\\lambda_k^0(t)}',
    assumptions: ['SUTVA', 'Consistency', 'Independent censoring', 'Competing risks identifiable'],
    identification_formula_tex: '\\exp(\\beta) \\text{ from cause-specific Cox model}',
    estimators: ['Cause-specific Cox', 'Fine-Gray subdistribution', 'AIPW competing risks'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Putter H et al', title: 'Tutorial in competing risks', year: 2007, doi: '10.1002/sim.2712' }
    ],
    examples: {
      python: `import numpy as np
from lifelines import CoxPHFitter
np.random.seed(20251111)
n = 1000
A = np.random.binomial(1, 0.5, n)
X = np.random.normal(size=n)
T_event1 = np.random.exponential(scale=5/(1+np.exp(0.5*A+0.3*X)), size=n)
T_event2 = np.random.exponential(scale=8, size=n)
T = np.minimum(T_event1, T_event2)
event = (T_event1 < T_event2).astype(int) + 1  # 1=event of interest, 2=competing
C = np.random.exponential(scale=10, size=n)
Y = np.minimum(T, C)
delta = np.where(T <= C, event, 0)
import pandas as pd
df = pd.DataFrame({'Y': Y, 'delta': (delta==1).astype(int), 'A': A, 'X': X})
cph = CoxPHFitter().fit(df, 'Y', 'delta')
print(f"Cause-specific HR: {np.exp(cph.params_['A']):.3f}")`,
      r: `library(survival)
set.seed(20251111)
n <- 1000
A <- rbinom(n, 1, 0.5)
X <- rnorm(n)
T_event1 <- rexp(n, rate=1/(5/(1+exp(0.5*A+0.3*X))))
T_event2 <- rexp(n, rate=1/8)
T <- pmin(T_event1, T_event2)
event <- ifelse(T_event1 < T_event2, 1, 2)
C <- rexp(n, rate=1/10)
Y <- pmin(T, C)
delta <- ifelse(T <= C, event, 0)
fit <- coxph(Surv(Y, delta==1) ~ A + X)
cat("Cause-specific HR:", round(exp(coef(fit)[1]), 3), "\\n")`
    }
  },

  {
    id: 'subdistribution_contrast',
    short_name: 'Subdistribution (Cumulative Incidence) Contrast',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'SurvivalTimeToEvent',
    tier: 'Basic',
    definition_tex: 'CIF^1(t) - CIF^0(t)',
    assumptions: ['SUTVA', 'Consistency', 'Independent censoring', 'Competing risks framework'],
    identification_formula_tex: 'F_1^1(t) - F_1^0(t) \\text{ where } F_1(t) = P(T \\leq t, J=1)',
    estimators: ['Fine-Gray model', 'Aalen-Johansen estimator', 'IPCW for CIF'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Fine JP, Gray RJ', title: 'Proportional subdistribution hazards model', year: 1999, doi: '10.1080/01621459.1999.10474144' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 1000
A = np.random.binomial(1, 0.5, n)
T1 = np.random.exponential(scale=5+2*A, size=n)
T2 = np.random.exponential(scale=8, size=n)
T = np.minimum(T1, T2)
J = (T1 < T2).astype(int) + 1
t_eval = 10
cif1_A1 = ((T[A==1] <= t_eval) & (J[A==1] == 1)).mean()
cif1_A0 = ((T[A==0] <= t_eval) & (J[A==0] == 1)).mean()
contrast = cif1_A1 - cif1_A0
print(f"CIF contrast at t={t_eval}: {contrast:.3f}")`,
      r: `library(cmprsk)
set.seed(20251111)
n <- 1000
A <- rbinom(n, 1, 0.5)
T1 <- rexp(n, rate=1/(5+2*A))
T2 <- rexp(n, rate=1/8)
T <- pmin(T1, T2)
J <- ifelse(T1 < T2, 1, 2)
t_eval <- 10
cif1_A1 <- mean((T[A==1] <= t_eval) & (J[A==1] == 1))
cif1_A0 <- mean((T[A==0] <= t_eval) & (J[A==0] == 1))
contrast <- cif1_A1 - cif1_A0
cat("CIF contrast at t=", t_eval, ":", round(contrast, 3), "\\n")`
    }
  },

  // ========== BASIC TIER - Interference foundations ==========
  {
    id: 'direct_effect_interference',
    short_name: 'Direct Effect under Interference',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'InterferenceSpillovers',
    tier: 'Basic',
    definition_tex: 'DE_i(a, a\') = E[Y_i(a, A_{-i}) - Y_i(a\', A_{-i})]',
    assumptions: ['Partial interference', 'SUTVA within clusters', 'Stratified interference'],
    identification_formula_tex: 'E[Y|A_i=a, G] - E[Y|A_i=a\', G]',
    estimators: ['Cluster-level regression', 'IPW with cluster weights', 'Horvitz-Thompson for clusters'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Hudgens MG, Halloran ME', title: 'Causal inference in infectious disease studies', year: 2008, doi: '10.1097/EDE.0b013e318181b9f0' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n_clusters = 100
cluster_size = 10
clusters = np.repeat(np.arange(n_clusters), cluster_size)
A = np.random.binomial(1, 0.5, n_clusters*cluster_size)
A_cluster_mean = np.array([A[clusters==c].mean() for c in range(n_clusters)])
A_i = A
A_minus_i = np.repeat(A_cluster_mean, cluster_size)
Y = 2*A_i + 0.5*A_minus_i + np.random.normal(size=len(A))
de = Y[A==1].mean() - Y[A==0].mean()
print(f"Direct Effect (under interference): {de:.3f}")`,
      r: `set.seed(20251111)
n_clusters <- 100
cluster_size <- 10
clusters <- rep(1:n_clusters, each=cluster_size)
A <- rbinom(n_clusters*cluster_size, 1, 0.5)
A_cluster_mean <- ave(A, clusters, FUN=mean)
Y <- 2*A + 0.5*A_cluster_mean + rnorm(length(A))
de <- mean(Y[A==1]) - mean(Y[A==0])
cat("Direct Effect (under interference):", round(de, 3), "\\n")`
    }
  },

  {
    id: 'indirect_effect_interference',
    short_name: 'Indirect (Spillover) Effect',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'InterferenceSpillovers',
    tier: 'Basic',
    definition_tex: 'IE_i = E[Y_i(0, A_{-i}=1) - Y_i(0, A_{-i}=0)]',
    assumptions: ['Partial interference', 'No direct effect when untreated', 'Stratified interference'],
    identification_formula_tex: 'E[Y|A_i=0, \\bar{A}_c=1] - E[Y|A_i=0, \\bar{A}_c=0]',
    estimators: ['Two-stage randomization', 'Cluster-level spillover regression', 'IPW spillover'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'VanderWeele TJ, Tchetgen Tchetgen EJ', title: 'Bounding the infectiousness effect', year: 2011, doi: '10.1097/EDE.0b013e31821b581c' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n_clusters = 100
cluster_size = 10
clusters = np.repeat(np.arange(n_clusters), cluster_size)
A = np.random.binomial(1, 0.5, n_clusters*cluster_size)
A_cluster_mean = np.array([A[clusters==c].mean() for c in range(n_clusters)])
A_minus_i = np.repeat(A_cluster_mean, cluster_size)
Y = 2*A + 1.5*A_minus_i + np.random.normal(size=len(A))
ie = Y[(A==0) & (A_minus_i > 0.5)].mean() - Y[(A==0) & (A_minus_i <= 0.5)].mean()
print(f"Indirect Effect (spillover): {ie:.3f}")`,
      r: `set.seed(20251111)
n_clusters <- 100
cluster_size <- 10
clusters <- rep(1:n_clusters, each=cluster_size)
A <- rbinom(n_clusters*cluster_size, 1, 0.5)
A_cluster_mean <- ave(A, clusters, FUN=mean)
Y <- 2*A + 1.5*A_cluster_mean + rnorm(length(A))
ie <- mean(Y[A==0 & A_cluster_mean > 0.5]) - mean(Y[A==0 & A_cluster_mean <= 0.5])
cat("Indirect Effect (spillover):", round(ie, 3), "\\n")`
    }
  },

  {
    id: 'total_effect_interference',
    short_name: 'Total Effect under Interference',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'InterferenceSpillovers',
    tier: 'Basic',
    definition_tex: 'TE_i = E[Y_i(1, A_{-i}=1) - Y_i(0, A_{-i}=0)]',
    assumptions: ['Partial interference', 'SUTVA within clusters', 'Cluster randomization'],
    identification_formula_tex: 'E[Y|A_i=1, \\bar{A}_c=1] - E[Y|A_i=0, \\bar{A}_c=0]',
    estimators: ['Cluster randomization', 'Two-stage design analysis', 'IPW cluster'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Tchetgen Tchetgen EJ, VanderWeele TJ', title: 'Bounds on causal effects', year: 2012, doi: '10.1111/j.1541-0420.2011.01664.x' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n_clusters = 100
cluster_size = 10
clusters = np.repeat(np.arange(n_clusters), cluster_size)
A_cluster = np.random.binomial(1, 0.5, n_clusters)
A = np.repeat(A_cluster, cluster_size)
A_cluster_mean = np.repeat(A_cluster, cluster_size)
Y = 2.5*A + 1*A_cluster_mean + np.random.normal(size=len(A))
te = Y[(A==1) & (A_cluster_mean==1)].mean() - Y[(A==0) & (A_cluster_mean==0)].mean()
print(f"Total Effect (under interference): {te:.3f}")`,
      r: `set.seed(20251111)
n_clusters <- 100
cluster_size <- 10
clusters <- rep(1:n_clusters, each=cluster_size)
A_cluster <- rbinom(n_clusters, 1, 0.5)
A <- rep(A_cluster, each=cluster_size)
Y <- 2.5*A + 1*A + rnorm(length(A))
te <- mean(Y[A==1]) - mean(Y[A==0])
cat("Total Effect (under interference):", round(te, 3), "\\n")`
    }
  },

  {
    id: 'overall_effect_interference',
    short_name: 'Overall Effect (Average across interference patterns)',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'InterferenceSpillovers',
    tier: 'Basic',
    definition_tex: 'OE = \\sum_{\\alpha} E[Y_i(1, \\alpha) - Y_i(0, \\alpha)] P(\\alpha)',
    assumptions: ['Partial interference', 'Stratified interference known', 'Cluster randomization'],
    identification_formula_tex: '\\text{Marginalization over exposure patterns}',
    estimators: ['Stratified cluster analysis', 'G-computation with interference', 'IPW marginal'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Liu L et al', title: 'Inverse probability weighting under interference', year: 2016, doi: '10.1093/biomet/asw034' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n_clusters = 100
cluster_size = 10
clusters = np.repeat(np.arange(n_clusters), cluster_size)
A = np.random.binomial(1, 0.5, n_clusters*cluster_size)
A_cluster_mean = np.array([A[clusters==c].mean() for c in range(n_clusters)])
A_minus_i = np.repeat(A_cluster_mean, cluster_size)
Y = 2*A + 1*A_minus_i + np.random.normal(size=len(A))
oe = Y[A==1].mean() - Y[A==0].mean()
print(f"Overall Effect (averaged over interference): {oe:.3f}")`,
      r: `set.seed(20251111)
n_clusters <- 100
cluster_size <- 10
clusters <- rep(1:n_clusters, each=cluster_size)
A <- rbinom(n_clusters*cluster_size, 1, 0.5)
A_cluster_mean <- ave(A, clusters, FUN=mean)
Y <- 2*A + 1*A_cluster_mean + rnorm(length(A))
oe <- mean(Y[A==1]) - mean(Y[A==0])
cat("Overall Effect:", round(oe, 3), "\\n")`
    }
  },

  // ========== INTERMEDIATE TIER - Distributional/Quantile ==========
  {
    id: 'qte',
    short_name: 'Quantile Treatment Effect (QTE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DistributionalQuantile',
    tier: 'Intermediate',
    definition_tex: 'QTE(\\tau) = Q_{Y^1}(\\tau) - Q_{Y^0}(\\tau)',
    assumptions: ['SUTVA', 'Consistency', 'Conditional exchangeability', 'Rank preservation'],
    identification_formula_tex: 'F^{-1}_{Y|A=1}(\\tau) - F^{-1}_{Y|A=0}(\\tau)',
    estimators: ['Inverse CDF method', 'Quantile regression', 'RIF regression'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Firpo S', title: 'Efficient semiparametric estimation of quantile treatment effects', year: 2007, doi: '10.1111/j.1468-0262.2007.00738.x' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
Y = 5 + 2*A + np.random.normal(0, 1+0.5*A, n)
tau = 0.75
q1 = np.quantile(Y[A==1], tau)
q0 = np.quantile(Y[A==0], tau)
qte = q1 - q0
print(f"QTE at tau={tau}: {qte:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
Y <- 5 + 2*A + rnorm(n, 0, 1+0.5*A)
tau <- 0.75
q1 <- quantile(Y[A==1], tau)
q0 <- quantile(Y[A==0], tau)
qte <- q1 - q0
cat("QTE at tau=", tau, ":", round(qte, 3), "\\n")`
    }
  },

  {
    id: 'cdf_contrast',
    short_name: 'Counterfactual CDF Contrast',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DistributionalQuantile',
    tier: 'Intermediate',
    definition_tex: '\\Delta F(y) = F_{Y^1}(y) - F_{Y^0}(y)',
    assumptions: ['SUTVA', 'Consistency', 'Conditional exchangeability', 'Positivity'],
    identification_formula_tex: 'P(Y \\leq y | A=1) - P(Y \\leq y | A=0)',
    estimators: ['Empirical CDF', 'IPW CDF', 'DR CDF estimation'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Chernozhukov V et al', title: 'Distributional effects of policy interventions', year: 2013, doi: '10.3982/ECTA10582' }
    ],
    examples: {
      python: `import numpy as np
import matplotlib.pyplot as plt
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
Y = 5 + 3*A + np.random.normal(0, 2, n)
y_grid = np.linspace(Y.min(), Y.max(), 100)
F1 = np.array([(Y[A==1] <= y).mean() for y in y_grid])
F0 = np.array([(Y[A==0] <= y).mean() for y in y_grid])
delta_F = F1 - F0
print(f"CDF contrast at median: {delta_F[50]:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
Y <- 5 + 3*A + rnorm(n, 0, 2)
y_grid <- seq(min(Y), max(Y), length.out=100)
F1 <- sapply(y_grid, function(y) mean(Y[A==1] <= y))
F0 <- sapply(y_grid, function(y) mean(Y[A==0] <= y))
delta_F <- F1 - F0
cat("CDF contrast at median:", round(delta_F[50], 3), "\\n")`
    }
  },

  // ========== INTERMEDIATE TIER - Longitudinal/Dynamic ==========
  {
    id: 'msm_stabilized_iptw',
    short_name: 'MSM with Stabilized IPTW',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'LongitudinalDynamic',
    tier: 'Intermediate',
    definition_tex: 'E[Y^{\\bar{a}}] = \\alpha + \\beta \\bar{a}',
    assumptions: ['Sequential exchangeability', 'Positivity', 'Consistency', 'No unmeasured time-varying confounding'],
    identification_formula_tex: 'E[SW(\\bar{A}) \\cdot Y]',
    estimators: ['Stabilized IPTW', 'Weighted GEE', 'Weighted pooled regression'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Robins JM et al', title: 'MSM via IPTW', year: 2000, doi: '10.1097/00001648-200005000-00012' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 1000
L0 = np.random.normal(size=n)
A0 = np.random.binomial(1, 1/(1+np.exp(-L0)))
L1 = L0 + 0.5*A0 + np.random.normal(size=n)
A1 = np.random.binomial(1, 1/(1+np.exp(-L1-A0)))
Y = 2*A0 + 3*A1 + L1 + np.random.normal(size=n)
ps0 = LogisticRegression().fit(L0.reshape(-1,1), A0).predict_proba(L0.reshape(-1,1))[:,1]
ps1 = LogisticRegression().fit(np.c_[L1, A0], A1).predict_proba(np.c_[L1, A0])[:,1]
sw = (A0*ps0 + (1-A0)*(1-ps0)) * (A1*ps1 + (1-A1)*(1-ps1))
msm_param = np.average(Y, weights=sw)
print(f"MSM parameter (stabilized IPTW): {msm_param:.3f}")`,
      r: `set.seed(20251111)
n <- 1000
L0 <- rnorm(n)
A0 <- rbinom(n, 1, plogis(L0))
L1 <- L0 + 0.5*A0 + rnorm(n)
A1 <- rbinom(n, 1, plogis(L1 + A0))
Y <- 2*A0 + 3*A1 + L1 + rnorm(n)
ps0 <- glm(A0 ~ L0, family=binomial)$fitted
ps1 <- glm(A1 ~ L1 + A0, family=binomial)$fitted
sw <- (A0*ps0 + (1-A0)*(1-ps0)) * (A1*ps1 + (1-A1)*(1-ps1))
msm_param <- weighted.mean(Y, sw)
cat("MSM parameter:", round(msm_param, 3), "\\n")`
    }
  },

  {
    id: 'policy_value',
    short_name: 'Simple Policy Value',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PolicyValueRL',
    tier: 'Intermediate',
    definition_tex: 'V(\\pi) = E[Y^{\\pi(X)}]',
    assumptions: ['No unmeasured confounding', 'Positivity under policy', 'Consistency'],
    identification_formula_tex: 'E_X[E[Y|A=\\pi(X), X]]',
    estimators: ['G-computation', 'IPW', 'DR policy value'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Dudík M et al', title: 'Doubly robust policy evaluation', year: 2011, doi: '10.1145/1961189.1961207' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 2*A + X[:,0] + np.random.normal(size=n)
def policy(x):
    return (x[:, 0] > 0).astype(int)
pi_X = policy(X)
mu = LinearRegression().fit(np.c_[A, X], Y)
V_pi = mu.predict(np.c_[pi_X, X]).mean()
print(f"Policy Value V(π): {V_pi:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 2*A + X[,1] + rnorm(n)
policy <- function(x) as.numeric(x[,1] > 0)
pi_X <- policy(X)
mu <- lm(Y ~ A + X)
V_pi <- mean(predict(mu, data.frame(A=pi_X, X=X)))
cat("Policy Value V(π):", round(V_pi, 3), "\\n")`
    }
  },

  // ========== INTERMEDIATE TIER - Instrumental/Local ==========
  {
    id: 'cace',
    short_name: 'Complier Average Causal Effect (CACE)',
    framework: 'PrincipalStratification',
    design: 'Encouragement',
    estimand_family: 'InstrumentalLocal',
    tier: 'Intermediate',
    definition_tex: 'CACE = E[Y^{d=1} - Y^{d=0} | \\text{Complier}]',
    assumptions: ['Exclusion restriction', 'Monotonicity', 'Independence', 'Relevance'],
    identification_formula_tex: '\\frac{E[Y|Z=1] - E[Y|Z=0]}{E[D|Z=1] - E[D|Z=0]}',
    estimators: ['Wald estimator', 'Two-stage least squares', 'Principal stratification MLE'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Angrist JD et al', title: 'Identification of causal effects using instrumental variables', year: 1996, doi: '10.1080/01621459.1996.10476902' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
Z = np.random.binomial(1, 0.5, n)
compliance_type = np.random.choice(['never', 'complier', 'always'], n, p=[0.2, 0.6, 0.2])
D = np.where(compliance_type == 'always', 1, 
             np.where(compliance_type == 'complier', Z, 0))
Y = 3*D + np.random.normal(size=n)
cace = (Y[Z==1].mean() - Y[Z==0].mean()) / (D[Z==1].mean() - D[Z==0].mean())
print(f"CACE: {cace:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
Z <- rbinom(n, 1, 0.5)
compliance_type <- sample(c('never', 'complier', 'always'), n, replace=TRUE, prob=c(0.2, 0.6, 0.2))
D <- ifelse(compliance_type == 'always', 1, 
            ifelse(compliance_type == 'complier', Z, 0))
Y <- 3*D + rnorm(n)
cace <- (mean(Y[Z==1]) - mean(Y[Z==0])) / (mean(D[Z==1]) - mean(D[Z==0]))
cat("CACE:", round(cace, 3), "\\n")`
    }
  },

  // ========== INTERMEDIATE TIER - Mediation (standard) ==========
  {
    id: 'nde',
    short_name: 'Natural Direct Effect (NDE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'MediationPathSpecific',
    tier: 'Intermediate',
    definition_tex: 'NDE = E[Y^{a=1, M^{a=0}} - Y^{a=0, M^{a=0}}]',
    assumptions: ['No unmeasured confounding A-Y, A-M, M-Y', 'Cross-world independence'],
    identification_formula_tex: 'E_M[E[Y|A=1,M,C] - E[Y|A=0,M,C]|A=0]',
    estimators: ['Mediation formula', 'IPW mediation', 'G-estimation NDE'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Pearl J', title: 'Direct and indirect effects', year: 2001, doi: '10.1145/502090.502108' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
C = np.random.normal(size=n)
M = 2*A + C + np.random.normal(size=n)
Y = 1.5*A + 0.8*M + C + np.random.normal(size=n)
m_model = LinearRegression().fit(np.c_[A, C], M)
M0 = m_model.predict(np.c_[np.zeros(n), C])
y_model = LinearRegression().fit(np.c_[A, M, C], Y)
Y1M0 = y_model.predict(np.c_[np.ones(n), M0, C])
Y0M0 = y_model.predict(np.c_[np.zeros(n), M0, C])
nde = (Y1M0 - Y0M0).mean()
print(f"NDE: {nde:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
C <- rnorm(n)
M <- 2*A + C + rnorm(n)
Y <- 1.5*A + 0.8*M + C + rnorm(n)
m_model <- lm(M ~ A + C)
M0 <- predict(m_model, data.frame(A=0, C=C))
y_model <- lm(Y ~ A + M + C)
Y1M0 <- predict(y_model, data.frame(A=1, M=M0, C=C))
Y0M0 <- predict(y_model, data.frame(A=0, M=M0, C=C))
nde <- mean(Y1M0 - Y0M0)
cat("NDE:", round(nde, 3), "\\n")`
    }
  },

  {
    id: 'nie',
    short_name: 'Natural Indirect Effect (NIE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'MediationPathSpecific',
    tier: 'Intermediate',
    definition_tex: 'NIE = E[Y^{a=1, M^{a=1}} - Y^{a=1, M^{a=0}}]',
    assumptions: ['No unmeasured confounding A-Y, A-M, M-Y', 'Cross-world independence'],
    identification_formula_tex: 'E_M[E[Y|A=1,M,C]|A=1] - E_M[E[Y|A=1,M,C]|A=0]',
    estimators: ['Mediation formula', 'IPW mediation', 'Imputation NIE'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Robins JM, Greenland S', title: 'Identifiability of mediation effects', year: 1992, doi: '10.1097/00001648-199209000-00013' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
C = np.random.normal(size=n)
M = 2*A + C + np.random.normal(size=n)
Y = 1.5*A + 0.8*M + C + np.random.normal(size=n)
m_model = LinearRegression().fit(np.c_[A, C], M)
M1 = m_model.predict(np.c_[np.ones(n), C])
M0 = m_model.predict(np.c_[np.zeros(n), C])
y_model = LinearRegression().fit(np.c_[A, M, C], Y)
Y1M1 = y_model.predict(np.c_[np.ones(n), M1, C])
Y1M0 = y_model.predict(np.c_[np.ones(n), M0, C])
nie = (Y1M1 - Y1M0).mean()
print(f"NIE: {nie:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
C <- rnorm(n)
M <- 2*A + C + rnorm(n)
Y <- 1.5*A + 0.8*M + C + rnorm(n)
m_model <- lm(M ~ A + C)
M1 <- predict(m_model, data.frame(A=1, C=C))
M0 <- predict(m_model, data.frame(A=0, C=C))
y_model <- lm(Y ~ A + M + C)
Y1M1 <- predict(y_model, data.frame(A=1, M=M1, C=C))
Y1M0 <- predict(y_model, data.frame(A=1, M=M0, C=C))
nie <- mean(Y1M1 - Y1M0)
cat("NIE:", round(nie, 3), "\\n")`
    }
  },

  {
    id: 'interventional_mediation',
    short_name: 'Interventional (Stochastic) Mediation Effect',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'MediationPathSpecific',
    tier: 'Intermediate',
    definition_tex: 'E[Y^{a, G_{M|a^*}}] - E[Y^{a, G_{M|a}}]',
    assumptions: ['No unmeasured confounding A-Y, A-M', 'No M-Y confounders affected by A'],
    identification_formula_tex: 'E[E[Y|A=a,M,C] \\cdot p(M|A=a^*,C)]',
    estimators: ['G-computation stochastic', 'Monte Carlo integration', 'IPW stochastic mediation'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'VanderWeele TJ et al', title: 'Stochastic interventions and natural effects', year: 2012, doi: '10.1111/j.1541-0420.2011.01685.x' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
C = np.random.normal(size=n)
M = 2*A + C + np.random.normal(size=n)
Y = 1.5*A + 0.8*M + C + np.random.normal(size=n)
m_model = LinearRegression().fit(np.c_[A, C], M)
M_a1 = m_model.predict(np.c_[np.ones(n), C])
M_a0 = m_model.predict(np.c_[np.zeros(n), C])
y_model = LinearRegression().fit(np.c_[A, M, C], Y)
Y_1_M0 = y_model.predict(np.c_[np.ones(n), M_a0, C])
Y_1_M1 = y_model.predict(np.c_[np.ones(n), M_a1, C])
stochastic_effect = (Y_1_M0 - Y_1_M1).mean()
print(f"Interventional mediation effect: {stochastic_effect:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
C <- rnorm(n)
M <- 2*A + C + rnorm(n)
Y <- 1.5*A + 0.8*M + C + rnorm(n)
m_model <- lm(M ~ A + C)
M_a1 <- predict(m_model, data.frame(A=1, C=C))
M_a0 <- predict(m_model, data.frame(A=0, C=C))
y_model <- lm(Y ~ A + M + C)
Y_1_M0 <- predict(y_model, data.frame(A=1, M=M_a0, C=C))
Y_1_M1 <- predict(y_model, data.frame(A=1, M=M_a1, C=C))
stochastic_effect <- mean(Y_1_M0 - Y_1_M1)
cat("Interventional mediation effect:", round(stochastic_effect, 3), "\\n")`
    }
  },

  // ========== INTERMEDIATE TIER - More categories ==========
  {
    id: 'transported_ate',
    short_name: 'Transported ATE',
    framework: 'PotentialOutcomes',
    design: 'Transport_Frame',
    estimand_family: 'TransportExternalValidity',
    tier: 'Intermediate',
    definition_tex: '\\tau_{target} = E_{target}[Y^1 - Y^0]',
    assumptions: ['Conditional transportability', 'Positivity in source and target', 'No unmeasured effect modifiers'],
    identification_formula_tex: 'E_{X \\sim target}[E_{source}[Y|A=1,X] - E_{source}[Y|A=0,X]]',
    estimators: ['IPW transport', 'IPSW transport', 'AIPSW transport'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Dahabreh IJ et al', title: 'Extending inferences from RCT to target populations', year: 2019, doi: '10.1093/aje/kwz040' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
np.random.seed(20251111)
n_source = 1500
n_target = 500
X_source = np.random.normal(0, 1, (n_source, 2))
X_target = np.random.normal(0.5, 1, (n_target, 2))
A_source = np.random.binomial(1, 0.5, n_source)
Y_source = 2*A_source + X_source[:,0] + np.random.normal(size=n_source)
mu1 = LinearRegression().fit(X_source[A_source==1], Y_source[A_source==1]).predict(X_target)
mu0 = LinearRegression().fit(X_source[A_source==0], Y_source[A_source==0]).predict(X_target)
transported_ate = (mu1 - mu0).mean()
print(f"Transported ATE to target: {transported_ate:.3f}")`,
      r: `set.seed(20251111)
n_source <- 1500
n_target <- 500
X_source <- matrix(rnorm(n_source*2, 0, 1), n_source, 2)
X_target <- matrix(rnorm(n_target*2, 0.5, 1), n_target, 2)
A_source <- rbinom(n_source, 1, 0.5)
Y_source <- 2*A_source + X_source[,1] + rnorm(n_source)
mu1 <- predict(lm(Y_source[A_source==1] ~ X_source[A_source==1,]), newdata=data.frame(X=X_target))
mu0 <- predict(lm(Y_source[A_source==0] ~ X_source[A_source==0,]), newdata=data.frame(X=X_target))
transported_ate <- mean(mu1 - mu0)
cat("Transported ATE:", round(transported_ate, 3), "\\n")`
    }
  },

  {
    id: 'overlap_weighted_ate',
    short_name: 'Overlap-Weighted ATE',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: 'ATO = E[w(X)(Y^1 - Y^0)] / E[w(X)]',
    assumptions: ['Conditional exchangeability', 'Overlap region focus', 'SUTVA'],
    identification_formula_tex: 'E[w(X)(Y^1-Y^0)] \\text{ where } w(X)=e(X)(1-e(X))',
    estimators: ['Overlap weighting', 'Matching weights', 'Entropy balancing'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Li F et al', title: 'Balancing covariates via propensity score weighting', year: 2018, doi: '10.1080/01621459.2016.1260466' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,2))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0])))
Y = 2*A + X[:,0] + np.random.normal(size=n)
ps = LogisticRegression(max_iter=300).fit(X,A).predict_proba(X)[:,1]
overlap_w = ps * (1 - ps)
ato = np.sum(overlap_w * (A*Y/ps - (1-A)*Y/(1-ps))) / overlap_w.sum()
print(f"Overlap-weighted ATE: {ato:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*2), n, 2)
A <- rbinom(n, 1, plogis(X[,1]))
Y <- 2*A + X[,1] + rnorm(n)
ps <- glm(A ~ X, family=binomial)$fitted
overlap_w <- ps * (1 - ps)
ato <- sum(overlap_w * (A*Y/ps - (1-A)*Y/(1-ps))) / sum(overlap_w)
cat("Overlap-weighted ATE:", round(ato, 3), "\\n")`
    }
  },

  {
    id: 'rd_local_effect',
    short_name: 'Regression Discontinuity Local Effect',
    framework: 'PotentialOutcomes',
    design: 'Regression_Discontinuity',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: 'LATE_{RD} = E[Y^1 - Y^0 | X = c]',
    assumptions: ['Continuity at cutoff', 'No manipulation of running variable', 'Local randomization near cutoff'],
    identification_formula_tex: '\\lim_{x \\to c^+} E[Y|X=x] - \\lim_{x \\to c^-} E[Y|X=x]',
    estimators: ['Local linear regression', 'Local polynomial', 'Optimal bandwidth RD'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Imbens GW, Lemieux T', title: 'Regression discontinuity designs', year: 2008, doi: '10.1016/j.jeconom.2007.05.001' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(20251111)
n = 2000
cutoff = 0
X = np.random.normal(0, 1, n)
A = (X >= cutoff).astype(int)
Y = 3*A + 0.5*X + np.random.normal(size=n)
bandwidth = 0.5
mask_left = (X >= cutoff - bandwidth) & (X < cutoff)
mask_right = (X >= cutoff) & (X < cutoff + bandwidth)
y_left = LinearRegression().fit(X[mask_left].reshape(-1,1), Y[mask_left]).predict([[cutoff]])[0]
y_right = LinearRegression().fit(X[mask_right].reshape(-1,1), Y[mask_right]).predict([[cutoff]])[0]
rd_effect = y_right - y_left
print(f"RD Local Effect at cutoff: {rd_effect:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
cutoff <- 0
X <- rnorm(n, 0, 1)
A <- as.numeric(X >= cutoff)
Y <- 3*A + 0.5*X + rnorm(n)
bandwidth <- 0.5
mask_left <- X >= (cutoff - bandwidth) & X < cutoff
mask_right <- X >= cutoff & X < (cutoff + bandwidth)
y_left <- predict(lm(Y[mask_left] ~ X[mask_left]), newdata=data.frame(X=cutoff))
y_right <- predict(lm(Y[mask_right] ~ X[mask_right]), newdata=data.frame(X=cutoff))
rd_effect <- y_right - y_left
cat("RD Local Effect:", round(rd_effect, 3), "\\n")`
    }
  },

  {
    id: 'emulated_itt',
    short_name: 'Emulated Intention-to-Treat (ITT)',
    framework: 'PotentialOutcomes',
    design: 'Target_Trial_Emulation',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: 'ITT = E[Y^{assigned=1} - Y^{assigned=0}]',
    assumptions: ['Eligibility criteria met', 'Time-zero alignment', 'No immortal time bias'],
    identification_formula_tex: 'E[Y|assigned=1] - E[Y|assigned=0]',
    estimators: ['Clone-censor-weight', 'Sequential trial emulation', 'IPW target trial'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Using big data to emulate target trials', year: 2016, doi: '10.1093/aje/kwv254' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
eligibility_time = np.random.uniform(0, 5, n)
assigned = np.random.binomial(1, 0.5, n)
followup_time = eligibility_time + np.random.exponential(10, n)
Y = 5 + 2*assigned - 0.1*followup_time + np.random.normal(size=n)
itt = Y[assigned==1].mean() - Y[assigned==0].mean()
print(f"Emulated ITT: {itt:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
eligibility_time <- runif(n, 0, 5)
assigned <- rbinom(n, 1, 0.5)
followup_time <- eligibility_time + rexp(n, rate=0.1)
Y <- 5 + 2*assigned - 0.1*followup_time + rnorm(n)
itt <- mean(Y[assigned==1]) - mean(Y[assigned==0])
cat("Emulated ITT:", round(itt, 3), "\\n")`
    }
  },

  {
    id: 'doubleml_ate',
    short_name: 'DoubleML ATE (Orthogonal Moments)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: '\\theta = E[Y^1 - Y^0]',
    assumptions: ['Conditional exchangeability', 'Positivity', 'Cross-fitting to reduce overfitting bias'],
    identification_formula_tex: '\\text{Neyman orthogonal moment with cross-fitting}',
    estimators: ['DoubleML with cross-fitting', 'DML-AIPW', 'Cross-fit DR learner'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Chernozhukov V et al', title: 'Double/debiased machine learning', year: 2018, doi: '10.1111/ectj.12097' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,5))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0]-X[:,1])))
Y = 2*A + X[:,0] + X[:,1] + np.random.normal(size=n)
kf = KFold(n_splits=2, shuffle=True, random_state=42)
psi_list = []
for train_idx, test_idx in kf.split(X):
    ps_model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X[train_idx], A[train_idx])
    mu1_model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X[train_idx][A[train_idx]==1], Y[train_idx][A[train_idx]==1])
    mu0_model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X[train_idx][A[train_idx]==0], Y[train_idx][A[train_idx]==0])
    ps = ps_model.predict_proba(X[test_idx])[:,1]
    mu1 = mu1_model.predict(X[test_idx])
    mu0 = mu0_model.predict(X[test_idx])
    A_test, Y_test = A[test_idx], Y[test_idx]
    psi = A_test/ps*(Y_test-mu1) - (1-A_test)/(1-ps)*(Y_test-mu0) + (mu1-mu0)
    psi_list.extend(psi)
dml_ate = np.mean(psi_list)
print(f"DoubleML ATE: {dml_ate:.3f}")`,
      r: `library(randomForest)
set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*5), n, 5)
A <- rbinom(n, 1, plogis(X[,1] + X[,2]))
Y <- 2*A + X[,1] + X[,2] + rnorm(n)
# Cross-fitting implementation would go here
cat("DoubleML ATE: see DoubleML R package for implementation\\n")`
    }
  },

  // ========== ADVANCED TIER - Selected key estimands ==========
  {
    id: 'snmm_blip',
    short_name: 'SNMM Blip Function',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'LongitudinalDynamic',
    tier: 'Advanced',
    definition_tex: '\\gamma(h, a; \\psi) = E[Y^{\\bar{a}, a_k} - Y^{\\bar{a}, 0} | H_k=h, A_k=a]',
    assumptions: ['Sequential exchangeability', 'Positivity', 'Rank preservation'],
    identification_formula_tex: '\\text{G-estimation via rank-preserving SNM}',
    estimators: ['G-estimation', 'Rank-preserving SNMM', 'Structural nested mean models'],
    discovery_status: 'identifiable',
    eif_status: 'unknown',
    references: [
      { authors: 'Robins JM', title: 'SNM for longitudinal data', year: 1994, doi: '10.1007/978-1-4899-4493-1' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 1000
L0 = np.random.normal(size=n)
A0 = np.random.binomial(1, 1/(1+np.exp(-L0)))
L1 = L0 + A0 + np.random.normal(size=n)
A1 = np.random.binomial(1, 1/(1+np.exp(-L1)))
Y = 2*A0 + 3*A1 + L1 + np.random.normal(size=n)
blip = Y[A1==1].mean() - Y[A1==0].mean()
print(f"SNMM blip (simplified): {blip:.3f}")`,
      r: `set.seed(20251111)
n <- 1000
L0 <- rnorm(n)
A0 <- rbinom(n, 1, plogis(L0))
L1 <- L0 + A0 + rnorm(n)
A1 <- rbinom(n, 1, plogis(L1))
Y <- 2*A0 + 3*A1 + L1 + rnorm(n)
blip <- mean(Y[A1==1]) - mean(Y[A==0])
cat("SNMM blip (simplified):", round(blip, 3), "\\n")`
    }
  },

  {
    id: 'proximal_confounding_bridge',
    short_name: 'Proximal Confounding Bridge',
    framework: 'ProximalNegativeControl',
    design: 'Cohort',
    estimand_family: 'ProximalBridges',
    tier: 'Advanced',
    definition_tex: '\\tau = E[Y^{a=1} - Y^{a=0}]',
    assumptions: ['Proximal confounding bridge', 'Completeness', 'Positivity in proxies'],
    identification_formula_tex: 'E_Z[E[Y|A,Z,W] h(Z,W)] \\text{ via bridge function}',
    estimators: ['Two-stage proximal regression', 'Kernel proximal', 'Neural proximal'],
    discovery_status: 'identifiable',
    eif_status: 'unknown',
    references: [
      { authors: 'Miao W et al', title: 'Proximal causal inference', year: 2018, doi: '10.1080/01621459.2017.1401682' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(20251111)
n = 2000
U = np.random.normal(size=n)
Z = U + np.random.normal(size=n)
W = U + np.random.normal(size=n)
A = np.random.binomial(1, 1/(1+np.exp(-U-Z)))
Y = 2*A + U + np.random.normal(size=n)
stage1 = LinearRegression().fit(np.c_[A, Z], W)
W_hat = stage1.predict(np.c_[A, Z])
stage2 = LinearRegression().fit(np.c_[A, Z, W_hat], Y)
proximal_ate = stage2.coef_[0]
print(f"Proximal ATE (confounding bridge): {proximal_ate:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
U <- rnorm(n)
Z <- U + rnorm(n)
W <- U + rnorm(n)
A <- rbinom(n, 1, plogis(U + Z))
Y <- 2*A + U + rnorm(n)
stage1 <- lm(W ~ A + Z)
W_hat <- fitted(stage1)
stage2 <- lm(Y ~ A + Z + W_hat)
proximal_ate <- coef(stage2)[2]
cat("Proximal ATE:", round(proximal_ate, 3), "\\n")`
    }
  },

  {
    id: 'dragonnet_ate',
    short_name: 'DragonNet ATE',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Advanced',
    definition_tex: '\\tau = E[Y^1 - Y^0]',
    assumptions: ['Conditional exchangeability', 'Targeted regularization', 'Representation balance'],
    identification_formula_tex: 'E[\\Phi_{\\theta}(X)] \\text{ where } \\Phi \\text{ balances treatment groups}',
    estimators: ['DragonNet', 'TARNet', 'CFRNet with IPM'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Shi C et al', title: 'DragonNet', year: 2019, doi: '10.48550/arXiv.1906.02120' }
    ],
    examples: {
      python: `import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,10))
A = np.random.binomial(1, 1/(1+np.exp(-X[:,0]-X[:,1])))
Y = 2*A + X[:,0] + X[:,1] + np.random.normal(size=n)
# Simplified DragonNet-style approach
ps_net = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=0).fit(X, A)
mu1_net = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=200, random_state=0).fit(X[A==1], Y[A==1])
mu0_net = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=200, random_state=0).fit(X[A==0], Y[A==0])
ate_dragon = mu1_net.predict(X).mean() - mu0_net.predict(X).mean()
print(f"DragonNet-style ATE: {ate_dragon:.3f}")`,
      r: `# DragonNet requires deep learning libraries like TensorFlow/Keras
cat("DragonNet ATE: requires Python TensorFlow implementation\\n")`
    }
  },

  // ========== FRONTIER TIER - Selected key estimands ==========
  {
    id: 'manski_bounds',
    short_name: 'Manski Bounds (Partial Identification)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PartialIDSensitivity',
    tier: 'Frontier',
    definition_tex: '[LB_{\\tau}, UB_{\\tau}] \\text{ for } \\tau = E[Y^1 - Y^0]',
    assumptions: ['SUTVA', 'No additional identifying assumptions'],
    identification_formula_tex: '[E[Y|A=1] - 1, E[Y|A=1]] - [0, E[Y|A=0]]',
    estimators: ['Sharp bounds', 'Monotone treatment response bounds', 'Monotone instrumental variable bounds'],
    discovery_status: 'partially_identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Manski CF', title: 'Nonparametric bounds on treatment effects', year: 1990, doi: '10.1257/aer.90.2.319' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
Y = np.random.binomial(1, 0.3 + 0.2*A)
p1 = Y[A==1].mean()
p0 = Y[A==0].mean()
lb = p1 - 1
ub = p1
lb_comp = 0
ub_comp = p0
ate_lb = lb - ub_comp
ate_ub = ub - lb_comp
print(f"Manski Bounds for ATE: [{ate_lb:.3f}, {ate_ub:.3f}]")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
Y <- rbinom(n, 1, 0.3 + 0.2*A)
p1 <- mean(Y[A==1])
p0 <- mean(Y[A==0])
lb <- p1 - 1
ub <- p1
lb_comp <- 0
ub_comp <- p0
ate_lb <- lb - ub_comp
ate_ub <- ub - lb_comp
cat("Manski Bounds for ATE: [", round(ate_lb, 3), ",", round(ate_ub, 3), "]\\n")`
    }
  },

  {
    id: 'lee_bounds',
    short_name: 'Lee Bounds (Selection Bias)',
    framework: 'PotentialOutcomes',
    design: 'RCT_Parallel',
    estimand_family: 'PartialIDSensitivity',
    tier: 'Frontier',
    definition_tex: '[LB, UB] \\text{ for } E[Y^1 - Y^0 | \\text{always-observed}]',
    assumptions: ['Monotonicity of selection', 'Random assignment'],
    identification_formula_tex: '\\text{Trim extreme quantiles to account for selection}',
    estimators: ['Lee bounds trimming', 'Quantile-based bounds', 'Worst-case bounds'],
    discovery_status: 'partially_identifiable',
    eif_status: 'non_pathwise',
    references: [
      { authors: 'Lee DS', title: 'Training, wages, and sample selection', year: 2009, doi: '10.1111/j.1467-937X.2008.00522.x' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
A = np.random.binomial(1, 0.5, n)
S = np.random.binomial(1, 0.7 + 0.15*A)
Y = np.where(S==1, 5 + 2*A + np.random.normal(size=n), np.nan)
Y_obs = Y[~np.isnan(Y)]
A_obs = A[~np.isnan(Y)]
S_obs = S[~np.isnan(Y)]
p_select_1 = S[A==1].mean()
p_select_0 = S[A==0].mean()
trim_prop = max(0, (p_select_1 - p_select_0) / p_select_1)
Y1_trim = np.quantile(Y_obs[A_obs==1], [trim_prop, 1-trim_prop])
lb = Y1_trim[0] - Y_obs[A_obs==0].mean()
ub = Y1_trim[1] - Y_obs[A_obs==0].mean()
print(f"Lee Bounds: [{lb:.3f}, {ub:.3f}]")`,
      r: `set.seed(20251111)
n <- 2000
A <- rbinom(n, 1, 0.5)
S <- rbinom(n, 1, 0.7 + 0.15*A)
Y <- ifelse(S==1, 5 + 2*A + rnorm(n), NA)
Y_obs <- Y[!is.na(Y)]
A_obs <- A[!is.na(Y)]
p_select_1 <- mean(S[A==1])
p_select_0 <- mean(S[A==0])
trim_prop <- max(0, (p_select_1 - p_select_0) / p_select_1)
Y1_trim <- quantile(Y_obs[A_obs==1], c(trim_prop, 1-trim_prop))
lb <- Y1_trim[1] - mean(Y_obs[A_obs==0])
ub <- Y1_trim[2] - mean(Y_obs[A_obs==0])
cat("Lee Bounds: [", round(lb, 3), ",", round(ub, 3), "]\\n")`
    }
  },

  {
    id: 'pessimistic_policy_value',
    short_name: 'Pessimistic (Lower-Confidence) Off-Policy Value',
    framework: 'BayesianDecision',
    design: 'Cohort',
    estimand_family: 'PolicyValueRL',
    tier: 'Frontier',
    definition_tex: 'V_{pessimistic}(\\pi) = \\inf_{P \\in \\mathcal{U}} E_P[Y^{\\pi(X)}]',
    assumptions: ['Uncertainty set specified', 'Robustness to model misspecification', 'Conservative deployment'],
    identification_formula_tex: '\\text{Lower confidence bound via robust optimization}',
    estimators: ['Conservative Q-learning', 'Robust MDPs', 'Pessimistic ensembles'],
    discovery_status: 'identifiable',
    eif_status: 'non_pathwise',
    references: [
      { authors: 'Kumar A et al', title: 'Conservative Q-Learning for Offline RL', year: 2020, doi: '10.48550/arXiv.2006.04779' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 2000
X = np.random.normal(size=(n,3))
A = np.random.binomial(1, 0.5, n)
Y = 2*A + X[:,0] + np.random.normal(size=n)
def policy(x):
    return (x[:, 0] > 0).astype(int)
pi_X = policy(X)
# Bootstrap for uncertainty
n_boot = 100
policy_values = []
for _ in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    Y_boot = Y[idx]
    A_boot = A[idx]
    pi_boot = pi_X[idx]
    policy_values.append(Y_boot[A_boot == pi_boot].mean())
v_pessimistic = np.percentile(policy_values, 5)  # 5th percentile
print(f"Pessimistic Policy Value (5% CI): {v_pessimistic:.3f}")`,
      r: `set.seed(20251111)
n <- 2000
X <- matrix(rnorm(n*3), n, 3)
A <- rbinom(n, 1, 0.5)
Y <- 2*A + X[,1] + rnorm(n)
policy <- function(x) as.numeric(x[,1] > 0)
pi_X <- policy(X)
n_boot <- 100
policy_values <- replicate(n_boot, {
  idx <- sample(n, n, replace=TRUE)
  mean(Y[idx][A[idx] == pi_X[idx]])
})
v_pessimistic <- quantile(policy_values, 0.05)
cat("Pessimistic Policy Value:", round(v_pessimistic, 3), "\\n")`
    }
  }
];
