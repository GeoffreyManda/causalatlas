export type Framework = 'PotentialOutcomes' | 'SCM' | 'PrincipalStratification' | 'ProximalNegativeControl' | 'BayesianDecision';
export type Design = 'RCT_Parallel' | 'Cluster_RCT' | 'Stepped_Wedge' | 'Factorial' | 'Encouragement' | 'Two_Stage' | 'Cohort' | 'Case_Control' | 'Cross_Sectional' | 'Case_Cohort' | 'SCCS' | 'Case_Crossover' | 'Regression_Discontinuity' | 'Natural_Experiment' | 'Target_Trial_Emulation' | 'Survey_Multistage' | 'Two_Phase' | 'Transport_Frame';
export type EstimandFamily = 'PopulationEffects' | 'DistributionalQuantile' | 'LongitudinalDynamic' | 'InstrumentalLocal' | 'MediationPathSpecific' | 'InterferenceSpillovers' | 'SurvivalTimeToEvent' | 'PartialIDSensitivity' | 'MissingnessMeasurementError' | 'ProximalBridges' | 'TransportExternalValidity' | 'PolicyValueRL' | 'DeepRepresentation';
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
  {
    id: 'att_cohort',
    short_name: 'ATT (Average Treatment Effect on Treated)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\psi = E[Y(1) - Y(0) | A=1]',
    assumptions: ['SUTVA', 'Exchangeability among treated', 'Positivity', 'Consistency'],
    identification_formula_tex: '\\psi = E[Y|A=1] - E_X[E[Y|A=0,X] | A=1]',
    estimators: ['Outcome regression', 'IPW-ATT', 'AIPW-ATT', 'Matching'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Rosenbaum & Rubin (1983)', 'Heckman et al. (1997)', 'Sant\'Anna & Zhao (2020)'],
  },
  {
    id: 'cate',
    short_name: 'CATE (Conditional ATE)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: '\\psi(x) = E[Y(1) - Y(0) | X=x]',
    assumptions: ['SUTVA', 'Conditional exchangeability', 'Positivity in subgroups', 'Consistency'],
    identification_formula_tex: '\\psi(x) = E[Y|A=1,X=x] - E[Y|A=0,X=x]',
    estimators: ['S-learner', 'T-learner', 'X-learner', 'R-learner', 'Causal forests'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Künzel et al. (2019)', 'Nie & Wager (2021)', 'Kennedy (2020)'],
  },
  {
    id: 'qte',
    short_name: 'QTE (Quantile Treatment Effect)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DistributionalQuantile',
    tier: 'Intermediate',
    definition_tex: '\\psi(\\tau) = F_{Y(1)}^{-1}(\\tau) - F_{Y(0)}^{-1}(\\tau)',
    assumptions: ['Rank invariance or exchangeability', 'Positivity', 'Consistency'],
    identification_formula_tex: '\\psi(\\tau) = Q_Y(\\tau|A=1) - Q_Y(\\tau|A=0)',
    estimators: ['Inverse rank weighting', 'Quantile regression', 'Distribution regression'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Firpo (2007)', 'Frölich & Melly (2013)', 'Chernozhukov et al. (2013)'],
  },
  {
    id: 'gformula',
    short_name: 'G-Formula (Longitudinal)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'LongitudinalDynamic',
    tier: 'Advanced',
    definition_tex: '\\psi = E[Y(\\bar{a})]',
    assumptions: ['Sequential exchangeability', 'Positivity over time', 'Consistency', 'No measurement error'],
    identification_formula_tex: '\\psi = \\int \\prod_t P(Y|\\bar{A}_t=\\bar{a}_t, \\bar{L}_t) P(L_t|\\bar{A}_{t-1}=\\bar{a}_{t-1}, \\bar{L}_{t-1}) d\\bar{L}',
    estimators: ['Parametric g-formula', 'Nonparametric g-formula', 'TMLE g-formula', 'Monte Carlo g-formula'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Robins (1986)', 'Hernán & Robins (2020)', 'Taubman et al. (2009)'],
  },
  {
    id: 'nde_nie',
    short_name: 'NDE/NIE (Natural Direct/Indirect Effects)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'MediationPathSpecific',
    tier: 'Advanced',
    definition_tex: '\\text{NDE} = E[Y(1,M(0)) - Y(0,M(0))], \\text{NIE} = E[Y(1,M(1)) - Y(1,M(0))]',
    assumptions: ['No unmeasured confounding (A-Y, A-M, M-Y)', 'No treatment-mediator interaction', 'Cross-world counterfactuals'],
    identification_formula_tex: '\\text{NDE} = E_M[E[Y|A=1,M,C] - E[Y|A=0,M,C] | A=0]',
    estimators: ['Mediation formula', 'IPW mediation', 'G-computation mediation', 'Natural effects model'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Pearl (2001)', 'VanderWeele (2015)', 'Tchetgen Tchetgen & Shpitser (2012)'],
  },
  {
    id: 'interference_direct',
    short_name: 'Direct Effect under Interference',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'InterferenceSpillovers',
    tier: 'Advanced',
    definition_tex: '\\psi = E[Y_i(1, A_{-i}) - Y_i(0, A_{-i})]',
    assumptions: ['Partial interference', 'Stratified interference', 'Neighborhood structure known'],
    identification_formula_tex: '\\psi = E[Y_i(1,\\alpha) - Y_i(0,\\alpha)]',
    estimators: ['IPW for interference', 'Horvitz-Thompson under spillovers', 'AIPW with exposure mapping'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Hudgens & Halloran (2008)', 'Tchetgen Tchetgen & VanderWeele (2012)', 'Aronow & Samii (2017)'],
  },
  {
    id: 'rmst',
    short_name: 'RMST Difference (Restricted Mean Survival Time)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'SurvivalTimeToEvent',
    tier: 'Intermediate',
    definition_tex: '\\psi(t^*) = \\int_0^{t^*} S_1(t) dt - \\int_0^{t^*} S_0(t) dt',
    assumptions: ['Independent censoring', 'Exchangeability', 'Positivity'],
    identification_formula_tex: '\\psi(t^*) = E[\\min(T(1), t^*)] - E[\\min(T(0), t^*)]',
    estimators: ['IPCW RMST', 'AIPW RMST', 'Pseudo-observation regression'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Uno et al. (2014)', 'Tian et al. (2014)', 'van der Laan & Hubbard (2005)'],
  },
  {
    id: 'manski_bounds',
    short_name: 'Manski Bounds',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'PartialIDSensitivity',
    tier: 'Advanced',
    definition_tex: '\\psi \\in [LB, UB] \\text{ with no assumptions}',
    assumptions: ['None (worst-case bounds)', 'Optional: Monotone treatment response', 'Optional: Monotone instrumental variable'],
    identification_formula_tex: 'LB = E[Y|A=1]\\cdot P(A=1) + y_{min}\\cdot P(A=0), UB = E[Y|A=1]\\cdot P(A=1) + y_{max}\\cdot P(A=0)',
    estimators: ['Plug-in bounds', 'Inference for partially identified parameters', 'Sensitivity analysis'],
    discovery_status: 'partially_identifiable',
    eif_status: 'non_pathwise',
    references: ['Manski (1990)', 'Manski (2003)', 'Imbens & Manski (2004)'],
  },
  {
    id: 'proximal_confounding',
    short_name: 'Proximal G-Formula',
    framework: 'ProximalNegativeControl',
    design: 'Cohort',
    estimand_family: 'ProximalBridges',
    tier: 'Frontier',
    definition_tex: '\\psi = E[Y(1) - Y(0)] \\text{ via proxies } Z, W',
    assumptions: ['Negative control outcome & treatment', 'Bridge function existence', 'Completeness conditions'],
    identification_formula_tex: '\\psi = E_W[E[Y|A=1,W] h(1,W) - E[Y|A=0,W] h(0,W)]',
    estimators: ['Proximal causal learning', 'Kernel bridge estimation', 'Semiparametric proximal inference'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Tchetgen Tchetgen et al. (2020)', 'Miao et al. (2018)', 'Cui et al. (2023)'],
  },
  {
    id: 'transported_ate',
    short_name: 'Transported ATE',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'TransportExternalValidity',
    tier: 'Advanced',
    definition_tex: '\\psi_{target} = E_{target}[Y(1) - Y(0)]',
    assumptions: ['Conditional exchangeability in trial', 'Transportability via X', 'Positivity in both populations'],
    identification_formula_tex: '\\psi_{target} = E_{target,X}[E_{trial}[Y|A=1,X] - E_{trial}[Y|A=0,X]]',
    estimators: ['IPSW (inverse probability of sampling weights)', 'Outcome model transport', 'Doubly robust transport'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Stuart et al. (2011)', 'Dahabreh et al. (2020)', 'Colnet et al. (2024)'],
  },
  {
    id: 'policy_value_ips',
    short_name: 'Policy Value (IPS)',
    framework: 'BayesianDecision',
    design: 'Cohort',
    estimand_family: 'PolicyValueRL',
    tier: 'Intermediate',
    definition_tex: '\\psi(\\pi) = E_{\\pi}[R] = E[\\frac{\\pi(A|X)}{\\pi_0(A|X)} R]',
    assumptions: ['Logging policy known', 'Positivity (overlap)', 'SUTVA'],
    identification_formula_tex: '\\psi(\\pi) = E[w(A,X) \\cdot R], w(A,X) = \\pi(A|X)/\\pi_0(A|X)',
    estimators: ['IPS', 'Weighted IPS', 'Doubly robust policy value', 'SWITCH estimator'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Dudík et al. (2014)', 'Kallus & Uehara (2020)', 'Su et al. (2020)'],
  },
  {
    id: 'cfr_net',
    short_name: 'CFRNet (Counterfactual Regression)',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Advanced',
    definition_tex: '\\psi = E[Y(1) - Y(0)] \\text{ via balanced } \\Phi(X)',
    assumptions: ['Representation sufficiency', 'Exchangeability in } \\Phi \\text{ space', 'Neural balancing'],
    identification_formula_tex: '\\psi = E_{\\Phi}[\\mu_1(\\Phi(X)) - \\mu_0(\\Phi(X))]',
    estimators: ['CFRNet with IPM balancing', 'Wasserstein CFR', 'TARNet variant'],
    discovery_status: 'identifiable',
    eif_status: 'unknown',
    references: ['Shalit et al. (2017)', 'Johansson et al. (2016)', 'Hassanpour & Greiner (2019)'],
  },
  {
    id: 'irm',
    short_name: 'IRM (Invariant Risk Minimization)',
    framework: 'SCM',
    design: 'Cohort',
    estimand_family: 'DeepRepresentation',
    tier: 'Frontier',
    definition_tex: '\\psi: \\min_\\Phi \\sum_e R_e(\\Phi) \\text{ s.t. } w^* \\text{ invariant across } e',
    assumptions: ['Multiple environments available', 'Causal features invariant', 'Spurious features vary'],
    identification_formula_tex: '\\Phi^* = \\arg\\min \\sum_e \\text{Risk}_e \\text{ with invariant predictor}',
    estimators: ['IRM penalty', 'IRMv1', 'Risk extrapolation methods'],
    discovery_status: 'partially_identifiable',
    eif_status: 'non_pathwise',
    references: ['Arjovsky et al. (2019)', 'Ahuja et al. (2020)', 'Rosenfeld et al. (2021)'],
  },
  {
    id: 'causal_discovery_pc',
    short_name: 'PC Algorithm (Causal Discovery)',
    framework: 'SCM',
    design: 'Cohort',
    estimand_family: 'PartialIDSensitivity',
    tier: 'Frontier',
    definition_tex: '\\text{Estimate DAG } \\mathcal{G} \\text{ from observational data}',
    assumptions: ['Causal sufficiency', 'Faithfulness', 'Causal Markov condition'],
    identification_formula_tex: '\\text{Learn skeleton via conditional independence, orient via v-structures}',
    estimators: ['PC', 'FCI', 'GES', 'NOTEARS', 'DAG-GNN'],
    discovery_status: 'partially_identifiable',
    eif_status: 'non_pathwise',
    references: ['Spirtes et al. (2000)', 'Zhang (2008)', 'Zheng et al. (2018)'],
  },
  {
    id: 'ate_rct_cluster',
    short_name: 'ATE (Cluster RCT)',
    framework: 'PotentialOutcomes',
    design: 'Cluster_RCT',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\psi = E[Y_i(1) - Y_i(0)]',
    assumptions: ['Cluster-level randomization', 'No interference between clusters', 'Consistency'],
    identification_formula_tex: '\\psi = E[\\bar{Y}_c | A_c=1] - E[\\bar{Y}_c | A_c=0]',
    estimators: ['Cluster-adjusted linear model', 'GEE', 'Mixed effects models', 'Cluster-robust SE'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Donner & Klar (2000)', 'Hayes & Moulton (2017)', 'Li et al. (2022)'],
  },
  {
    id: 'did_modern',
    short_name: 'DiD with Staggered Adoption',
    framework: 'PotentialOutcomes',
    design: 'Cohort',
    estimand_family: 'LongitudinalDynamic',
    tier: 'Advanced',
    definition_tex: '\\psi_{g,t} = E[Y_{g,t}(1) - Y_{g,t}(0) | G=g]',
    assumptions: ['Parallel trends', 'No anticipation', 'Homogeneous treatment effects (or aggregate with weights)'],
    identification_formula_tex: '\\psi = \\text{Aggregation of group-time ATTs with honest weights}',
    estimators: ['Callaway-Sant\'Anna', 'Sun-Abraham', 'Borusyak-Hull-Jaravel', 'Stacked DiD'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Callaway & Sant\'Anna (2021)', 'Sun & Abraham (2021)', 'Borusyak et al. (2024)'],
  },
  {
    id: 'case_control_or',
    short_name: 'Odds Ratio (Case-Control)',
    framework: 'PotentialOutcomes',
    design: 'Case_Control',
    estimand_family: 'PopulationEffects',
    tier: 'Basic',
    definition_tex: '\\psi = \\frac{P(A=1|D=1)/P(A=0|D=1)}{P(A=1|D=0)/P(A=0|D=0)}',
    assumptions: ['Rare disease approximation', 'No selection bias conditional on A,X', 'Correct matching/sampling'],
    identification_formula_tex: '\\psi = \\text{OR from logistic model or stratified analysis}',
    estimators: ['Conditional logistic regression', 'Mantel-Haenszel OR', 'Bayesian case-control models'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Breslow & Day (1980)', 'Rothman et al. (2008)', 'Greenland & Thomas (1982)'],
  },
  {
    id: 'sccs',
    short_name: 'SCCS (Self-Controlled Case Series)',
    framework: 'PotentialOutcomes',
    design: 'Case_Crossover',
    estimand_family: 'PopulationEffects',
    tier: 'Intermediate',
    definition_tex: '\\psi = \\text{IRR within individual risk periods}',
    assumptions: ['Event does not censor observation', 'Exposure changes over time', 'No time-varying confounding within individual'],
    identification_formula_tex: '\\psi = \\frac{\\lambda(t|\\text{exposed})}{\\lambda(t|\\text{unexposed})}',
    estimators: ['Conditional Poisson regression', 'Age-adjusted SCCS', 'SCCS with time-varying covariates'],
    discovery_status: 'identifiable',
    eif_status: 'available',
    references: ['Farrington (1995)', 'Whitaker et al. (2006)', 'Petersen et al. (2016)'],
  },
];
