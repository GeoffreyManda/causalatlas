export type Lang = 'python' | 'r';

export interface EstimandRequirements {
  python: string[];
  r: string[];
}

// Minimal per-estimand requirements. Fall back to defaults when missing.
const defaults: EstimandRequirements = {
  python: ['micropip', 'numpy', 'pandas', 'scipy', 'scikit-learn'],
  r: ['stats']
};

const reqs: Record<string, Partial<EstimandRequirements>> = {
  ate: {
    python: ['numpy', 'pandas', 'scikit-learn'],
    r: ['stats']
  },
  att: {
    python: ['numpy', 'scikit-learn'],
    r: ['stats']
  },
  cate: {
    python: ['numpy', 'scikit-learn'],
    r: ['stats']
  },
  qte: {
    python: ['numpy'],
    r: ['stats']
  },
  msm: {
    python: ['numpy', 'scikit-learn'],
    r: ['stats']
  },
  late: {
    python: ['numpy'],
    r: ['stats']
  },
  nde: {
    python: ['numpy'],
    r: ['stats']
  },
  rmst: {
    python: ['numpy'],
    r: ['survival']
  },
  interference_direct: {
    python: ['numpy'],
    r: ['stats']
  },
  manski_bounds: {
    python: ['numpy'],
    r: ['stats']
  },
  proximal_gformula: {
    python: ['numpy', 'pandas'],
    r: ['stats']
  },
  bayesian_ate: {
    python: ['numpy'],
    r: ['stats']
  },
  bayesian_sensitivity: {
    python: ['numpy'],
    r: ['stats']
  },
  value_of_information: {
    python: ['numpy'],
    r: ['stats']
  }
};

export function getRequirements(estimandId: string): EstimandRequirements {
  const entry = reqs[estimandId] || {};
  return {
    python: entry.python ?? defaults.python,
    r: entry.r ?? defaults.r
  };
}
