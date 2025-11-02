import { Framework, Design, EstimandFamily } from './estimands';

export interface DataStructureContent {
  requiredVariables: string[];
  timeStructure: string;
  additionalVariables: string[];
  compatibleDesigns: string[];
  sampleSizeConsiderations: string;
}

export interface DiagnosticsContent {
  positivityChecks: string[];
  balanceAssessment: string[];
  modelChecks: string[];
  residualDiagnostics: string[];
}

export interface SensitivityContent {
  unmeasuredConfounding: string[];
  measurementError: string[];
  modelMisspecification: string[];
  transportability: string[];
}

export interface EthicsContent {
  targetTrialProtocol: string[];
  dataProvenance: string[];
  biasRegistry: string[];
  reproducibility: string[];
}

interface SlideContent {
  dataStructure: DataStructureContent;
  diagnostics: DiagnosticsContent;
  sensitivity: SensitivityContent;
  ethics: EthicsContent;
}

// Content for ATE (Average Treatment Effect)
const ateContent: SlideContent = {
  dataStructure: {
    requiredVariables: [
      'Outcome Y (continuous or binary)',
      'Treatment A (binary: 0=control, 1=treated)',
      'Confounders X (all common causes of A and Y)',
      'Optional: Effect modifiers V'
    ],
    timeStructure: 'Cross-sectional: X measured before A, A before Y',
    additionalVariables: [
      'Mediators M (post-treatment, if mediation analysis planned)',
      'Negative controls (for bias detection)',
      'Instrumental variables (if available)'
    ],
    compatibleDesigns: [
      'Randomized Controlled Trials (RCT)',
      'Prospective cohort studies',
      'Retrospective cohort with rich covariates',
      'Target trial emulation',
      'Natural experiments with conditional randomization'
    ],
    sampleSizeConsiderations: 'For 80% power to detect effect δ: n ≈ 16σ²/δ² per arm. Adjust for covariate imbalance and effect heterogeneity.'
  },
  diagnostics: {
    positivityChecks: [
      'Plot propensity score distributions by treatment group',
      'Check overlap: min(p) > 0.05 and max(p) < 0.95',
      'Identify regions of covariate space with sparse treatment representation',
      'Trim extreme propensity scores or use bounded weights'
    ],
    balanceAssessment: [
      'Standardized mean differences (SMD < 0.1 after weighting)',
      'Variance ratios close to 1.0',
      'QQ-plots of weighted vs unweighted distributions',
      'Love plots showing balance improvement'
    ],
    modelChecks: [
      'Cross-validated prediction accuracy for outcome models',
      'Hosmer-Lemeshow test for propensity score calibration',
      'Super Learner ensemble to reduce model dependence',
      'Residual plots for outcome regression models'
    ],
    residualDiagnostics: [
      'Plot influence function (IF) residuals by covariates',
      'Check IF mean ≈ 0 and symmetry',
      'Identify high-leverage observations',
      'Bootstrap confidence intervals for uncertainty quantification'
    ]
  },
  sensitivity: {
    unmeasuredConfounding: [
      'E-value: minimum strength of unmeasured confounder to explain away effect',
      'Rosenbaum Γ sensitivity bounds (for matched studies)',
      'Ding-VanderWeele sensitivity analysis framework',
      'Negative control outcomes to detect residual confounding'
    ],
    measurementError: [
      'Regression calibration if validation data available',
      'SIMEX (Simulation-Extrapolation) for error correction',
      'Sensitivity to outcome misclassification (for binary Y)',
      'Multiple imputation for missing data under MAR/MNAR'
    ],
    modelMisspecification: [
      'Compare parametric vs nonparametric estimators',
      'Augmented IPW (AIPW) for double robustness',
      'Cross-fit nuisance parameter estimation',
      'Assess sensitivity to bandwidth/tuning parameter choices'
    ],
    transportability: [
      'Assess covariate distribution differences between populations',
      'Inverse odds weighting for external validity',
      'Stratified analysis by site or population subgroups',
      'Effect modification analysis to identify non-transportable effects'
    ]
  },
  ethics: {
    targetTrialProtocol: [
      'Eligibility criteria: clearly define target population',
      'Treatment assignment: specify start/stop rules',
      'Outcomes: primary and secondary, with timing',
      'Follow-up: duration and censoring mechanisms',
      'Causal contrast: per-protocol vs intention-to-treat'
    ],
    dataProvenance: [
      'Document data source (EHR, claims, survey)',
      'Consent type (broad, specific, waived for secondary use)',
      'Linkage procedures and privacy protections',
      'Exclusions and missing data patterns',
      'Pre-registration or analysis plan timestamp'
    ],
    biasRegistry: [
      'Selection bias: immortal time, prevalent user, depletion of susceptibles',
      'Information bias: measurement error, recall bias',
      'Confounding bias: unmeasured confounding severity',
      'Quantify bias direction and magnitude where possible'
    ],
    reproducibility: [
      'Random seeds for all stochastic procedures',
      'Software versions (R, Python, package versions)',
      'Computational environment (Docker/Conda hash)',
      'Audit logs: data access, analysis modifications, results',
      'Code repository with version control (Git SHA)'
    ]
  }
};

// Shared content templates for different estimand families
export const getSlideContentByFamily = (family: EstimandFamily, framework: Framework, design: Design, estimandId: string): SlideContent => {
  // Start with ATE template and customize
  const baseContent = JSON.parse(JSON.stringify(ateContent)) as SlideContent;

  // Customize for specific estimand families
  switch (family) {
    case 'InstrumentalLocal':
      baseContent.dataStructure.requiredVariables.push(
        'Instrumental variable Z (affects A but not Y except through A)',
        'Compliance indicators (for LATE estimation)'
      );
      baseContent.dataStructure.additionalVariables.push(
        'Always-takers, never-takers, compliers (compliance strata)',
        'Interaction between Z and baseline covariates'
      );
      baseContent.diagnostics.modelChecks.push(
        'F-statistic > 10 for first-stage regression',
        'Sargan test for overidentification (if multiple instruments)',
        'Bound sensitivity to monotonicity violations'
      );
      break;

    case 'MediationPathSpecific':
      baseContent.dataStructure.additionalVariables.push(
        'Mediator M measured post-treatment, pre-outcome',
        'Treatment-mediator interactions if present',
        'Post-treatment confounders L (if present, may violate cross-world independence)'
      );
      baseContent.dataStructure.timeStructure = 'Longitudinal: X → A → M → Y';
      baseContent.diagnostics.modelChecks.push(
        'Check mediation proportion bounds (natural effects)',
        'Sensitivity to cross-world independence violations',
        'Assess intermediate confounding'
      );
      baseContent.sensitivity.unmeasuredConfounding.push(
        'No unmeasured A-Y confounding',
        'No unmeasured M-Y confounding',
        'No unmeasured A-M confounding',
        'No M-Y confounder affected by A (cross-world independence)'
      );
      break;

    case 'SurvivalTimeToEvent':
      baseContent.dataStructure.requiredVariables.push(
        'Event time T or censoring time C',
        'Event indicator δ = I(T ≤ C)',
        'Time-varying confounders L(t) if present'
      );
      baseContent.dataStructure.timeStructure = 'Longitudinal with censoring: L(t) → A(t) → Y(T)';
      baseContent.diagnostics.modelChecks.push(
        'Kaplan-Meier curves by treatment',
        'Schoenfeld residuals for proportional hazards',
        'Time-varying coefficient checks',
        'Censoring mechanism validation (MCAR/MAR)'
      );
      break;

    case 'LongitudinalDynamic':
      baseContent.dataStructure.timeStructure = 'Sequential: L₀ → A₀ → L₁ → A₁ → ... → Y';
      baseContent.dataStructure.requiredVariables.push(
        'Time-varying treatments A(t)',
        'Time-varying confounders L(t)',
        'Treatment history Ā(t)'
      );
      baseContent.diagnostics.modelChecks.push(
        'Check positivity violations over time',
        'Assess weight stability (IPW)',
        'Evaluate g-formula fit at each time point'
      );
      baseContent.sensitivity.unmeasuredConfounding.push(
        'Sequential exchangeability at each time t',
        'Positivity at each time t',
        'No unmeasured time-varying confounding',
        'Consistency for dynamic strategies'
      );
      break;

    case 'InterferenceSpillovers':
      baseContent.dataStructure.requiredVariables.push(
        'Cluster/network identifiers',
        'Individual treatment A_i and cluster treatment Ā_c',
        'Spillover exposure (e.g., proportion treated in network)'
      );
      baseContent.diagnostics.modelChecks.push(
        'Check interference mechanism (spillover functional form)',
        'Assess cluster size variation',
        'Evaluate correlation structure within clusters'
      );
      baseContent.sensitivity.unmeasuredConfounding.push(
        'Partial interference: interference only within clusters',
        'Stratified exchangeability within clusters',
        'Cluster-level positivity'
      );
      break;

    case 'DistributionalQuantile':
      baseContent.dataStructure.requiredVariables.push(
        'Full outcome distribution Y (not just mean)',
        'Quantile of interest τ ∈ (0,1)'
      );
      baseContent.diagnostics.modelChecks.push(
        'Quantile regression diagnostics',
        'Check crossing of conditional quantile functions',
        'Bootstrap for quantile treatment effect CI'
      );
      break;

    case 'ProximalBridges':
      baseContent.dataStructure.requiredVariables.push(
        'Proximal treatment confounder W (mediates hidden confounding of A)',
        'Proximal outcome confounder Z (mediates hidden confounding of Y)',
        'Hidden confounder U (unobserved)'
      );
      baseContent.diagnostics.modelChecks.push(
        'Validate bridge function estimates',
        'Check proxy relevance (correlations with treatment/outcome)',
        'Sensitivity to completeness violations'
      );
      baseContent.sensitivity.unmeasuredConfounding.push(
        'W-completeness: W contains all A-relevant info about U',
        'Z-completeness: Z contains all Y-relevant info about U',
        'Bridge equations solvable (proximal identification)'
      );
      break;

    case 'TransportExternalValidity':
      baseContent.dataStructure.requiredVariables.push(
        'Selection indicator S (1=study sample, 0=target population)',
        'Variables predictive of selection S',
        'Effect modifiers that differ between study and target'
      );
      baseContent.diagnostics.modelChecks.push(
        'Compare covariate distributions across S=0 and S=1',
        'Check positivity of selection scores',
        'Assess effect heterogeneity by key covariates'
      );
      break;
  }

  return baseContent;
};

// Export default content
export const estimandSlideContent: Record<string, SlideContent> = {
  ate: ateContent,
  att: {
    ...ateContent,
    dataStructure: {
      ...ateContent.dataStructure,
      sampleSizeConsiderations: 'Focus on treated group: ensure sufficient control observations with similar covariate distributions.'
    }
  },
  // Add more specific overrides as needed
};
