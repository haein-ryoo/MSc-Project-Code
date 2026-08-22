data {
  int<lower=4> T;
  vector[T] f;
}

parameters {
  real alpha;
  real<lower=0, upper=1> phi;
  real<lower=0.001, upper=0.5> sigma1;
}

model {
  alpha ~ normal(-0.1, 0.1);
  phi ~ normal(0.5, 0.2);

  for (t in 3:T) {
    real delta_t = f[t] - f[t - 1];
    real delta_lag = f[t - 1] - f[t - 2];
    delta_t ~ normal(alpha + phi * delta_lag, sigma1);
  }
}

generated quantities {
  vector[T - 2] log_lik;
  vector[T] f_rep;

  f_rep[1] = f[1];
  f_rep[2] = f[2];

  for (t in 3:T) {
    real delta_t = f[t] - f[t - 1];
    real delta_lag = f[t - 1] - f[t - 2];
    log_lik[t - 2] = normal_lpdf(delta_t | alpha + phi * delta_lag, sigma1);
  }

  for (t in 3:T) {
    real delta_lag_rep = f_rep[t - 1] - f_rep[t - 2];
    f_rep[t] = f_rep[t - 1]
               + normal_rng(alpha + phi * delta_lag_rep, sigma1);
  }
}
