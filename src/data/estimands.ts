export type Framework = 'PotentialOutcomes' | 'SCM' | 'PrincipalStratification' | 'ProximalNegativeControl' | 'BayesianDecision';
export type Design = 'RCT_Parallel' | 'Cohort' | 'Case_Control' | 'Regression_Discontinuity' | 'Natural_Experiment';
export type EstimandFamily = 'PopulationEffects' | 'DistributionalQuantile' | 'LongitudinalDynamic' | 'InstrumentalLocal' | 'MediationPathSpecific' | 'InterferenceSpillovers' | 'SurvivalTimeToEvent' | 'DeepRepresentation';
export type Tier = 'Basic' | 'Intermediate' | 'Advanced' | 'Frontier';
export type DiscoveryStatus = 'identifiable' | 'partially_identifiable' | 'non_identifiable';
export type EIFStatus = 'available' | 'unknown' | 'non_pathwise';

export interface CodeExample {
  language: 'Python' | 'R';
  code: string;
  seed: number;
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
  references: string[];
  examples?: CodeExample[];
}

export const estimandsData: Estimand[] = [
  {
    id: 'ate_rct',
    short_name: 'ATE (RCT)',
    framework: 'PotentialOutcomes',
    design: 'RCT_Parallel',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\psi = E[Y(1) - Y(0)]',
    assumptions: ['SUTVA', 'Random assignment', 'Consistency'],
    identification_formula_tex: '\\psi = E[Y|A=1] - E[Y|A=0]',
    estimators: ['Difference in means', 'ANCOVA', 'IPW', 'AIPW/DR'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Neyman (1923)', 'Rubin (1974)', 'Robins et al. (1994)'],
    examples: [{
      language: 'Python',
      seed: 20251111,
      code: `import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
np.random.seed(20251111)
n=2000
X=np.random.normal(size=(n,3))
e=1/(1+np.exp(-(0.3*X[:,0]-0.2*X[:,1]+0.1*X[:,2])))
A=np.random.binomial(1,e)
Y=2*A + X[:,0] + 0.5*X[:,1] + np.random.normal(size=n)
idx=np.arange(n); np.random.shuffle(idx); folds=[idx[:n//2], idx[n//2:]]
phi=np.zeros(n)
for k in range(2):
    train=folds[1-k]; test=folds[k]
    prop=LogisticRegression(max_iter=300).fit(X[train],A[train])
    ehat=prop.predict_proba(X[test])[:,1]
    mu1=LinearRegression().fit(X[train][A[train]==1], Y[train][A[train]==1])
    mu0=LinearRegression().fit(X[train][A[train]==0], Y[train][A[train]==0])
    mu1h=mu1.predict(X[test]); mu0h=mu0.predict(X[test])
    Ai=A[test]; Yi=Y[test]
    phi[test]=Ai/ehat*(Yi-mu1h)-(1-Ai)/(1-ehat)*(Yi-mu0h)+(mu1h-mu0h)
psi=phi.mean(); se=phi.std(ddof=1)/np.sqrt(n)
print(f"ATE (AIPW): {psi:.3f} ± {1.96*se:.3f}")`
    }]
  },
  {
    id: 'ate_cohort',
    short_name: 'ATE (Cohort)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\psi = E[Y(1) - Y(0)]',
    assumptions: ['SUTVA', 'Exchangeability (no unmeasured confounding)', 'Positivity', 'Consistency'],
    identification_formula_tex: '\\psi = E_X[E[Y|A=1,X] - E[Y|A=0,X]]',
    estimators: ['G-computation', 'IPW', 'AIPW/DR', 'TMLE', 'DoubleML'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Robins (1986)', 'van der Laan & Rubin (2006)', 'Hernán & Robins (2020)'],
  },
  {
    id: 'doubleml_ate',
    short_name: 'DoubleML ATE',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Intermediate',
    definition_tex: '\\psi = E[Y(1) - Y(0)]',
    assumptions: ['Exchangeability', 'Positivity', 'ML nuisance convergence', 'Orthogonality'],
    identification_formula_tex: '\\psi = \\arg\\min_{\\theta} E[(A - e(X))(Y - m(X) - \\theta A)]',
    estimators: ['Cross-fitted moment equations', 'DR/R/X-learners', 'Debiased ML'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Chernozhukov et al. (2018)', 'Kennedy (2020)'],
    examples: [{
      language: 'Python',
      seed: 20251111,
      code: `import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegression
np.random.seed(20251111)
n=3000; p=20
X=np.random.normal(size=(n,p))
true_e = 1/(1+np.exp(-X[:,0]-0.5*X[:,1]+0.2*X[:,2]))
A=np.random.binomial(1,true_e)
Y=1.5*A + X[:,0] + 0.5*X[:,1] + np.random.normal(size=n)
y_hat = LassoCV(cv=3).fit(X,Y).predict(X)
a_hat = LogisticRegression(max_iter=300).fit(X,A).predict_proba(X)[:,1]
y_res = Y - y_hat
a_res = A - a_hat
tau = np.dot(a_res, y_res) / np.dot(a_res, a_res)
print(f"DoubleML ATE: {tau:.3f}")`
    }]
  },
  {
    id: 'dragonnet',
    short_name: 'DragonNet Representation',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Advanced',
    definition_tex: '\\psi = E[Y(1) - Y(0)] \\text{ via balanced } \\Phi(X)',
    assumptions: ['Exchangeability in representation space', 'Positivity', 'Neural architecture sufficiency'],
    identification_formula_tex: '\\psi = E_{\\Phi}[E[Y|A=1,\\Phi(X)] - E[Y|A=0,\\Phi(X)]]',
    estimators: ['DragonNet', 'CFRNet', 'BALD', 'TARNet with balancing'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Shi et al. (2019)', 'Shalit et al. (2017)', 'Künzel et al. (2019)'],
  },
  {
    id: 'rd_local',
    short_name: 'RD Local Effect',
    framework: 'SCM',
    design: 'Regression_Discontinuity',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: '\\psi = \\lim_{z \\downarrow c} E[Y|Z=z] - \\lim_{z \\uparrow c} E[Y|Z=z]',
    assumptions: ['Continuity of E[Y(a)|Z] at cutoff', 'No manipulation', 'Local randomization'],
    identification_formula_tex: '\\psi = E[Y(1) - Y(0) | Z=c]',
    estimators: ['Local linear regression', 'Local polynomial', 'Robust bias-corrected'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Hahn et al. (2001)', 'Imbens & Lemieux (2008)', 'Calonico et al. (2014)'],
    examples: [{
      language: 'R',
      seed: 20251111,
      code: `set.seed(20251111)
n <- 5000
Z <- runif(n,-1,1)
A <- as.integer(Z >= 0)
Y <- 2*A + 0.5*Z + rnorm(n,0,0.5)
library(locfit)
bw <- 0.2
fit_pos <- locfit(Y ~ lp(Z), subset=Z>=0 & abs(Z)<bw)
fit_neg <- locfit(Y ~ lp(Z), subset=Z<0 & abs(Z)<bw)
tau <- predict(fit_pos, newdata=data.frame(Z=0)) - 
       predict(fit_neg, newdata=data.frame(Z=0))
print(paste("RD Effect:", round(tau, 3)))`
    }]
  },
  {
    id: 'msm_longitudinal',
    short_name: 'MSM (Marginal Structural Model)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'LongitudinalDynamic',
    tier: 'Intermediate',
    definition_tex: '\\psi: E[Y(\\bar{a})] = \\beta_0 + \\beta_1 a_1 + \\beta_2 a_2',
    assumptions: ['Sequential exchangeability', 'Positivity over time', 'Correct MSM specification'],
    identification_formula_tex: '\\psi \\text{ via IPW: } E[Y \\cdot \\prod_t \\frac{P(A_t=a_t)}{P(A_t|\\bar{L}_t)}]',
    estimators: ['IPW-MSM', 'G-estimation', 'TMLE for MSM', 'DR-MSM'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Robins (1997)', 'Hernán et al. (2000)', 'van der Laan & Petersen (2007)'],
    examples: [{
      language: 'R',
      seed: 20251111,
      code: `set.seed(20251111)
n <- 3000
L1 <- rnorm(n); A1 <- rbinom(n,1,plogis(0.5*L1))
L2 <- 0.4*A1 + rnorm(n); A2 <- rbinom(n,1,plogis(0.5*L2 - 0.3*A1))
Y <- 1.2*A1 + 1.5*A2 + 0.5*L1 + 0.3*L2 + rnorm(n)
pA1 <- plogis(0.5*L1); pA2 <- plogis(0.5*L2 - 0.3*A1)
sw <- (plogis(0.5*A1)*plogis(0.5*A2))/(pA1*pA2)
msm <- glm(Y ~ A1 + A2, weights=sw)
print(coef(msm))`
    }]
  },
  {
    id: 'km_survival',
    short_name: 'Survival Difference (KM)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'SurvivalTimeToEvent',
    tier: 'Basic',
    definition_tex: '\\psi(t) = S_1(t) - S_0(t) = P(T(1) > t) - P(T(0) > t)',
    assumptions: ['Independent censoring', 'Exchangeability', 'Positivity'],
    identification_formula_tex: '\\psi(t) = P(T > t | A=1) - P(T > t | A=0)',
    estimators: ['Kaplan-Meier difference', 'IPTW Kaplan-Meier', 'TMLE for survival'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Kaplan & Meier (1958)', 'Tsiatis (2006)', 'van der Laan & Robins (2003)'],
    examples: [{
      language: 'Python',
      seed: 20251111,
      code: `import numpy as np
np.random.seed(20251111)
n=3000
X=np.random.normal(size=(n,2))
A=np.random.binomial(1, 1/(1+np.exp(-0.4*X[:,0])))
base=np.random.exponential(scale=1.0,size=n)
T=base/(1+0.6*A); C=np.random.exponential(scale=2.0,size=n)
time=np.minimum(T,C); event=(T<=C).astype(int)
def km(time,event,t=1.0):
    order=np.argsort(time); at=len(time); s=1.0
    for i in order:
        if event[i]==1: s *= (at-1)/at
        at -= 1
        if time[i]>t: break
    return s
s1=km(time[A==1],event[A==1]); s0=km(time[A==0],event[A==0])
print(f"KM survival diff at t=1: {s1-s0:.3f}")`
    }]
  },
  {
    id: 'late_iv',
    short_name: 'LATE (Local Average Treatment Effect)',
    framework: 'PrincipalStratification',
    design: 'Natural_Experiment',
    estimand_family: 'InstrumentalLocal',
    tier: 'Intermediate',
    definition_tex: '\\psi = E[Y(1) - Y(0) | \\text{Complier}]',
    assumptions: ['Instrument independence', 'Exclusion restriction', 'Monotonicity', 'First stage'],
    identification_formula_tex: '\\psi = \\frac{E[Y|Z=1] - E[Y|Z=0]}{E[A|Z=1] - E[A|Z=0]}',
    estimators: ['Wald estimator', '2SLS', 'GMM', 'Doubly robust IV'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Imbens & Angrist (1994)', 'Angrist et al. (1996)', 'Abadie (2003)'],
  },
];
