functions {
  // Conditional log likelihood for one candidate Phase III entry point.
  real candidate_log_lik(vector f, int tau,
                         real alpha, real phi, real sigma1,
                         real mu, real rho, real sigma2) {
    int T = num_elements(f);
    real lp = 0;

    // Phase II: AR(1) process for first differences, conditional on Delta f[2].
    for (t in 3:tau) {
      real delta_t = f[t] - f[t - 1];
      real delta_lag = f[t - 1] - f[t - 2];
      lp += normal_lpdf(delta_t | alpha + phi * delta_lag, sigma1);
    }

    // Phase III: stationary, mean-reverting AR(1) process for TFR levels.
    for (t in (tau + 1):T) {
      lp += normal_lpdf(f[t] | (1 - rho) * mu + rho * f[t - 1], sigma2);
    }

    return lp;
  }
}

data {
  int<lower=4> T;
  vector[T] f;                         // observed annual TFR

  int<lower=1> N_tau;
  array[N_tau] int<lower=3, upper=T - 1> tau_candidates;

  // 1 = inflated UN empirical Bayes, 2 = reference,
  // 3 = penalised complexity.
  int<lower=1, upper=3> prior_type;

  // Empirical Bayes summaries from the UN hierarchical model.
  real mu_bar;
  real<lower=0> sigma_mu;
  real<lower=0, upper=1> rho_bar;
  real<lower=0> sigma_rho;

  // Rate of the exponential PC prior on distance from rho = 0.
  real<lower=0> lambda_pc;
}

parameters {
  // Phase II nuisance parameters.
  real alpha;
  real<lower=0, upper=1> phi;
  real<lower=0.001, upper=0.5> sigma1;

  // Phase III parameters of inferential interest.
  real<lower=0, upper=2.1> mu;
  real<lower=0.001, upper=0.999> rho;
  real<lower=0.001, upper=0.5> sigma2;
}

model {
  vector[N_tau] lp_tau;

  // Fixed Phase II prior: this component is nuisance-only in the comparison.
  alpha ~ normal(-0.1, 0.1);
  phi ~ normal(0.5, 0.2);
  // Uniform priors for sigma1 and, except below, sigma2 are induced by bounds.

  if (prior_type == 1) {
    // Inflated UN prior: original bayesTFR empirical Bayes SDs multiplied by 2
    // to reduce prior-data conflict for the South Korean series.
    mu ~ normal(mu_bar, 2 * sigma_mu);
    rho ~ normal(rho_bar, 2 * sigma_rho);
  } else if (prior_type == 2) {
    // Reference prior: pi(mu, rho, sigma2) proportional to (1-rho)/sigma2.
    target += log1m(rho) - log(sigma2);
  } else {
    real distance = sqrt(-log1m(square(rho)));
    real log_abs_jacobian = log(rho)
                            - log1m(square(rho))
                            - log(distance);

    mu ~ normal(mu_bar, sigma_mu);
    // PC prior induced by distance = sqrt(-log(1-rho^2)).
    target += log(lambda_pc) - lambda_pc * distance + log_abs_jacobian;
  }

  // Uniform p(tau) = 1/N_tau; log_sum_exp analytically marginalises tau.
  for (k in 1:N_tau) {
    lp_tau[k] = candidate_log_lik(f, tau_candidates[k],
                                  alpha, phi, sigma1,
                                  mu, rho, sigma2)
                - log(N_tau);
  }
  target += log_sum_exp(lp_tau);
}

generated quantities {
  vector[N_tau] log_lik_by_tau;
  simplex[N_tau] tau_prob;
  int<lower=1, upper=N_tau> tau_index;
  int tau_draw;
  vector[T - 2] log_lik;
  vector[T] f_rep;
  vector[10] f_forecast;

  for (k in 1:N_tau) {
    log_lik_by_tau[k] = candidate_log_lik(f, tau_candidates[k],
                                          alpha, phi, sigma1,
                                          mu, rho, sigma2);
  }

  // Conditional p(tau_k | f, theta) for this HMC draw.
  tau_prob = softmax(log_lik_by_tau);
  tau_index = categorical_rng(tau_prob);
  tau_draw = tau_candidates[tau_index];

  // Conditional observation-level log scores under the sampled tau.
  // These support an approximate PSIS-LOO diagnostic. Since tau is a single
  // global latent variable, this is not exact marginalized pointwise LOO.
  for (t in 3:T) {
    if (t <= tau_draw) {
      real delta_t = f[t] - f[t - 1];
      real delta_lag = f[t - 1] - f[t - 2];
      log_lik[t - 2] = normal_lpdf(delta_t |
                                      alpha + phi * delta_lag, sigma1);
    } else {
      log_lik[t - 2] = normal_lpdf(f[t] |
                                      (1 - rho) * mu + rho * f[t - 1], sigma2);
    }
  }

  // Replicate a complete trajectory conditional on the first two observations.
  f_rep[1] = f[1];
  f_rep[2] = f[2];
  for (t in 3:tau_draw) {
    real delta_lag_rep = f_rep[t - 1] - f_rep[t - 2];
    f_rep[t] = f_rep[t - 1]
               + normal_rng(alpha + phi * delta_lag_rep, sigma1);
  }
  for (t in (tau_draw + 1):T) {
    f_rep[t] = normal_rng((1 - rho) * mu + rho * f_rep[t - 1], sigma2);
  }

  // Posterior predictive Phase III trajectory from the final observation.
  f_forecast[1] = normal_rng((1 - rho) * mu + rho * f[T], sigma2);
  for (h in 2:10) {
    f_forecast[h] = normal_rng((1 - rho) * mu
                               + rho * f_forecast[h - 1], sigma2);
  }
}
