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
    id: 'intro_causality',
    short_name: 'Introduction to Causal Inference',
    framework: 'PotentialOutcomes',
    design: 'RCT_Parallel',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\text{Counterfactual: } Y^a \\text{ vs } Y^{a\'}',
    assumptions: ['Counterfactual existence', 'Well-defined interventions'],
    identification_formula_tex: 'E[Y^a] - E[Y^{a\'}]',
    estimators: ['Difference in means'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: [
      { authors: 'Hernán MA, Robins JM', title: 'Causal Inference: What If', year: 2020, doi: '10.1201/9781420013542' }
    ],
    examples: {
      python: `import numpy as np
np.random.seed(20251111)
n = 1000
A = np.random.binomial(1, 0.5, n)
Y = 2*A + np.random.normal(0, 1, n)
ATE = Y[A==1].mean() - Y[A==0].mean()
print(f"ATE from RCT: {ATE:.3f}")`,
      r: `set.seed(20251111)
n <- 1000
A <- rbinom(n, 1, 0.5)
Y <- 2*A + rnorm(n)
ATE <- mean(Y[A==1]) - mean(Y[A==0])
cat("ATE:", round(ATE, 3), "\\n")`
    }
  },
  
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
  }
];
