// [[Rcpp::depends(RcppArmadillo, mSSL)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <mSSL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

using namespace Rcpp;
using namespace arma;
using namespace linconGaussR;
using namespace quic;

namespace {

constexpr double BB_EPS = 1e-12;

inline double clamp_probability(const double x) {
  return std::max(1e-8, std::min(1.0 - 1e-8, x));
}

inline double stable_log_sum_exp(const double a, const double b) {
  const double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

inline double slab_probability(const double value,
                               const double mixing,
                               const double slab_rate,
                               const double spike_rate) {
  const double log_odds_spike =
    std::log1p(-mixing) - std::log(mixing) +
    std::log(spike_rate) - std::log(slab_rate) -
    std::abs(value) * (spike_rate - slab_rate);

  if (log_odds_spike > 35.0) return 0.0;
  if (log_odds_spike < -35.0) return 1.0;
  return 1.0 / (1.0 + std::exp(log_odds_spike));
}

inline double bb_objective(const double n_eff,
                           const int p,
                           const int q,
                           const arma::mat& S,
                           const arma::mat& centered_B,
                           const arma::mat& Omega,
                           const double lambda1,
                           const double lambda0,
                           const double xi1,
                           const double xi0,
                           const double theta,
                           const double eta,
                           const int diag_penalty,
                           const arma::vec& theta_hyper_params,
                           const arma::vec& eta_hyper_params) {
  double log_det_value = 0.0;
  double log_det_sign = 0.0;
  arma::log_det(log_det_value, log_det_sign, Omega);
  if (log_det_sign <= 0.0 || !std::isfinite(log_det_value)) {
    return -std::numeric_limits<double>::infinity();
  }

  double out = 0.5 * n_eff *
    (log_det_value - arma::trace(S * Omega));

  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < q; ++k) {
      const double value = std::abs(centered_B(j, k));
      const double slab = std::log(theta) + std::log(lambda1) - lambda1 * value;
      const double spike = std::log1p(-theta) + std::log(lambda0) - lambda0 * value;
      out += stable_log_sum_exp(slab, spike);
    }
  }

  for (int k = 0; k < q; ++k) {
    for (int kk = k + 1; kk < q; ++kk) {
      const double value = std::abs(Omega(k, kk));
      const double slab = std::log(eta) + std::log(xi1) - xi1 * value;
      const double spike = std::log1p(-eta) + std::log(xi0) - xi0 * value;
      out += stable_log_sum_exp(slab, spike);
    }
  }

  if (diag_penalty == 1) {
    for (int k = 0; k < q; ++k) {
      out += std::log(xi1) - xi1 * std::abs(Omega(k, k));
    }
  }

  const double a_theta = theta_hyper_params(0);
  const double b_theta = theta_hyper_params(1);
  const double a_eta = eta_hyper_params(0);
  const double b_eta = eta_hyper_params(1);

  out += (a_theta - 1.0) * std::log(theta) +
    (b_theta - 1.0) * std::log1p(-theta);
  out += (a_eta - 1.0) * std::log(eta) +
    (b_eta - 1.0) * std::log1p(-eta);
  return out;
}

inline void rescale_binary_covariance(arma::mat& Sigma,
                                      arma::mat& Omega,
                                      const int q_binary) {
  if (q_binary <= 0) return;

  arma::vec full_diagonal = Sigma.diag();
  arma::vec scaling = arma::sqrt(
    full_diagonal.subvec(0, static_cast<arma::uword>(q_binary - 1))
  );

  if (!scaling.is_finite() || arma::any(scaling <= 0.0)) {
    Rcpp::stop("Non-positive binary latent variance encountered.");
  }

  Omega.rows(0, q_binary - 1).each_col() %= scaling;
  Omega.cols(0, q_binary - 1).each_row() %= scaling.t();

  scaling = 1.0 / scaling;
  Sigma.rows(0, q_binary - 1).each_col() %= scaling;
  Sigma.cols(0, q_binary - 1).each_row() %= scaling.t();
}

class BBMixedWorkingParam {
 public:
  arma::mat X;
  arma::mat Y;
  arma::vec weights;
  arma::mat center;
  arma::mat S;
  arma::mat R;
  arma::mat tXX;
  arma::mat tXR;
  arma::mat tRR;
  arma::vec mu;
  arma::vec s_eval;
  int q_binary;
  double n_eff;

  BBMixedWorkingParam(const arma::mat& X_,
                      const arma::mat& Y_,
                      const arma::vec& weights_,
                      const arma::mat& center_,
                      const int q_binary_)
    : X(X_), Y(Y_), weights(weights_), center(center_),
      q_binary(q_binary_), n_eff(arma::accu(weights_)) {
    const int q = Y.n_cols;
    const int p = X.n_cols;
    S.zeros(q, q);
    R.zeros(Y.n_rows, q);
    tXX.zeros(p, p);
    tXR.zeros(p, q);
    tRR.zeros(q, q);
    mu = (Y.t() * weights) / n_eff;
    s_eval.zeros(q);
  }

  void update(const arma::vec& mu_t,
              const arma::mat& centered_B,
              const arma::mat& Sigma_t,
              const int n_rep,
              const int nskp) {
    const int n = Y.n_rows;
    const int q_total = Y.n_cols;
    const int q_continuous = q_total - q_binary;

    const arma::mat actual_B = centered_B + center;
    const arma::mat XB_no_intercept = X * actual_B;
    arma::mat XB = XB_no_intercept;
    XB.each_row() += mu_t.t();

    R.zeros(n, q_total);
    mu.zeros(q_total);

    if (q_continuous > 0) {
      const int first_continuous = q_binary;
      for (int k = first_continuous; k < q_total; ++k) {
        mu(k) = arma::dot(weights, Y.col(k) - XB_no_intercept.col(k)) / n_eff;
        R.col(k) = Y.col(k) - XB.col(k);
      }
    }

    if (q_binary > 0) {
      arma::mat binary_cov = Sigma_t.submat(
        0, 0, q_binary - 1, q_binary - 1
      );
      arma::mat chol_cov;
      if (!arma::chol(chol_cov, binary_cov)) {
        Rcpp::stop("Cholesky factorization failed for the binary latent covariance.");
      }

      arma::vec weighted_binary_mean(q_binary, arma::fill::zeros);
      arma::mat binary_residual_mean(n, q_binary, arma::fill::zeros);

      for (int i = 0; i < n; ++i) {
        Rcpp::checkUserInterrupt();

        arma::vec signs =
          2.0 * arma::trans(Y.row(i).cols(0, q_binary - 1)) - 1.0;
        arma::vec boundary =
          arma::trans(XB.row(i).cols(0, q_binary - 1));
        arma::mat constraints = chol_cov.t();
        constraints.each_col() %= signs;
        boundary %= signs;

        arma::vec initial = arma::solve(
          constraints,
          0.001 - boundary
        );
        LinearConstraints lincon(constraints, boundary, true);
        EllipticalSliceSampler sampler(n_rep + 1, lincon, nskp, initial);
        sampler.run();

        arma::mat residual_draws = sampler.loop_state.samples;
        residual_draws.shed_row(0);
        residual_draws = residual_draws * chol_cov;

        const arma::rowvec residual_mean = arma::mean(residual_draws, 0);
        binary_residual_mean.row(i) = residual_mean;
        weighted_binary_mean += weights(i) *
          arma::trans(residual_mean + mu_t.subvec(0, q_binary - 1).t());
      }

      R.cols(0, q_binary - 1) = binary_residual_mean;
      mu.subvec(0, q_binary - 1) = weighted_binary_mean / n_eff;
    }

    // Convert residuals formed using mu_t to residuals formed using the
    // newly updated (weighted) intercept.
    R.each_row() += mu_t.t();
    R.each_row() -= mu.t();

    arma::mat weighted_R = R;
    weighted_R.each_col() %= weights;
    arma::mat weighted_X = X;
    weighted_X.each_col() %= weights;

    tXR = X.t() * weighted_R;
    tXX = X.t() * weighted_X;
    tRR = R.t() * weighted_R;
    S = tRR / n_eff;
    s_eval = arma::eig_sym(S);
  }
};

inline void bb_coordinate_descent(const double n_eff,
                                  const int p,
                                  const int q,
                                  arma::mat& centered_B,
                                  arma::mat& R,
                                  arma::mat& tXR,
                                  arma::mat& S,
                                  const double theta,
                                  const arma::mat& Omega,
                                  const arma::mat& X,
                                  const arma::mat& tXX,
                                  const double lambda1,
                                  const double lambda0,
                                  const int max_iter,
                                  const double eps,
                                  const int verbose) {
  arma::umat active = arma::conv_to<arma::umat>::from(centered_B != 0.0);
  int iter = 0;
  int violations = 0;
  bool converged = false;

  while (iter < max_iter) {
    converged = false;

    while (iter < max_iter) {
      ++iter;
      converged = true;

      for (int j = 0; j < p; ++j) {
        const double curvature = std::max(tXX(j, j), BB_EPS);
        for (int k = 0; k < q; ++k) {
          if (active(j, k) == 0u) continue;

          const double omega_kk = std::max(Omega(k, k), BB_EPS);
          const double old_value = centered_B(j, k);
          const double score =
            arma::dot(tXR.row(j), Omega.row(k)) / omega_kk +
            curvature * old_value;

          double new_value = 0.0;
          if (lambda0 == lambda1) {
            new_value = std::copysign(
              std::max(std::abs(score) - lambda1 / omega_kk, 0.0) /
                curvature,
              score
            );
          } else {
            const double pstar0 = slab_probability(
              0.0, theta, lambda1, lambda0
            );
            const double pstar = slab_probability(
              old_value, theta, lambda1, lambda0
            );
            const double lambda_star0 =
              lambda1 * pstar0 + lambda0 * (1.0 - pstar0);
            const double lambda_star =
              lambda1 * pstar + lambda0 * (1.0 - pstar);
            const double log_inverse_pstar0 = -std::log(pstar0);
            const double g =
              std::pow(lambda_star0 - lambda1, 2.0) -
              2.0 * curvature * omega_kk * log_inverse_pstar0;

            double threshold = lambda_star0 / omega_kk;
            if (g > 0.0 &&
                (lambda0 - lambda1) >
                  std::sqrt(curvature) / (2.0 * std::sqrt(omega_kk))) {
              threshold =
                std::sqrt(
                  2.0 * curvature * log_inverse_pstar0 / omega_kk
                ) + lambda1 / omega_kk;
            }

            if (std::abs(score) > threshold) {
              new_value = std::copysign(
                std::max(std::abs(score) - lambda_star / omega_kk, 0.0) /
                  curvature,
                score
              );
            }
          }

          const double shift = old_value - new_value;
          centered_B(j, k) = new_value;
          R.col(k) += X.col(j) * shift;
          S.row(k) += shift * tXR.row(j) / n_eff;
          S.col(k) += shift * tXR.row(j).t() / n_eff;
          S(k, k) += tXX(j, j) * shift * shift / n_eff;
          tXR.col(k) += tXX.col(j) * shift;

          if (std::abs(shift) > eps * (1.0 + std::abs(old_value))) {
            converged = false;
          }
        }
      }

      if (converged) break;
    }

    violations = 0;
    for (int j = 0; j < p; ++j) {
      const double curvature = std::max(tXX(j, j), BB_EPS);
      for (int k = 0; k < q; ++k) {
        if (active(j, k) != 0u) continue;

        const double omega_kk = std::max(Omega(k, k), BB_EPS);
        const double score =
          arma::dot(tXR.row(j), Omega.row(k)) / omega_kk;
        double new_value = 0.0;

        if (lambda0 == lambda1) {
          new_value = std::copysign(
            std::max(std::abs(score) - lambda1 / omega_kk, 0.0) /
              curvature,
            score
          );
        } else {
          const double pstar0 = slab_probability(
            0.0, theta, lambda1, lambda0
          );
          const double lambda_star0 =
            lambda1 * pstar0 + lambda0 * (1.0 - pstar0);
          const double log_inverse_pstar0 = -std::log(pstar0);
          const double g =
            std::pow(lambda_star0 - lambda1, 2.0) -
            2.0 * curvature * omega_kk * log_inverse_pstar0;

          double threshold = lambda_star0 / omega_kk;
          if (g > 0.0 &&
              (lambda0 - lambda1) >
                std::sqrt(curvature) / (2.0 * std::sqrt(omega_kk))) {
            threshold =
              std::sqrt(
                2.0 * curvature * log_inverse_pstar0 / omega_kk
              ) + lambda1 / omega_kk;
          }

          if (std::abs(score) > threshold) {
            new_value = std::copysign(
              std::max(std::abs(score) - lambda_star0 / omega_kk, 0.0) /
                curvature,
              score
            );
          }
        }

        if (new_value != 0.0) {
          ++violations;
          const double shift = -new_value;
          centered_B(j, k) = new_value;
          R.col(k) += X.col(j) * shift;
          S.row(k) += shift * tXR.row(j) / n_eff;
          S.col(k) += shift * tXR.row(j).t() / n_eff;
          S(k, k) += tXX(j, j) * shift * shift / n_eff;
          tXR.col(k) += tXX.col(j) * shift;
        }
      }
    }

    active = arma::conv_to<arma::umat>::from(centered_B != 0.0);
    if (violations == 0) break;
  }

  if (iter >= max_iter && (!converged || violations != 0) && verbose == 1) {
    Rcpp::Rcout << "    [BB-SSL B update] maximum iterations reached; "
                << violations << " inactive-set violation(s) remain.\n";
  }
}

inline void bb_update_B_theta(const double n_eff,
                              const int p,
                              const int q,
                              arma::mat& centered_B,
                              arma::mat& R,
                              arma::mat& tXR,
                              arma::mat& S,
                              double& theta,
                              const arma::mat& Omega,
                              const arma::mat& X,
                              const arma::mat& tXX,
                              const double lambda1,
                              const double lambda0,
                              const arma::vec& theta_hyper_params,
                              const int max_iter,
                              const double eps,
                              const int verbose) {
  const int coordinate_max_iter = std::max(5 * max_iter, 50);

  for (int iter = 0; iter < max_iter; ++iter) {
    const arma::mat old_B = centered_B;

    bb_coordinate_descent(
      n_eff, p, q, centered_B, R, tXR, S, theta, Omega, X, tXX,
      lambda1, lambda0, coordinate_max_iter, eps, verbose
    );

    mSSL::update_theta(
      static_cast<int>(std::round(n_eff)), p, q, theta, centered_B,
      lambda1, lambda0, theta_hyper_params
    );
    theta = clamp_probability(theta);

    const arma::umat support_changed =
      arma::conv_to<arma::umat>::from((old_B != 0.0) != (centered_B != 0.0));
    const double relative_change =
      arma::norm(centered_B - old_B, "fro") /
      std::max(1.0, arma::norm(old_B, "fro"));

    if (arma::accu(support_changed) == 0u && relative_change <= eps) break;
  }
}

}  // namespace


//' One fixed-penalty weighted mixed-mSSL fit for BB-SSL
//'
//' This is the compiled engine used by bb_mixed_mssl(). Observation weights
//' are normalized in R before entry. The SSL prior for B is centered at the
//' supplied random target matrix, while the graphical SSL prior remains
//' centered at zero.
//'
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List bb_mpcssl_fixed_cpp(
    arma::mat X,
    const arma::mat& Y,
    const int binidxend,
    const arma::vec& weights,
    const arma::mat& center_std,
    const double lambda1,
    const double lambda0,
    const double xi1,
    const double xi0,
    const arma::vec& theta_hyper_params,
    const arma::vec& eta_hyper_params,
    const int diag_penalty,
    const int max_iter,
    const double eps,
    const double s_max_condition,
    const int obj_counter_max,
    const int verbose,
    const int nrep,
    const int nskp,
    const arma::mat& initial_B,
    const arma::mat& initial_Omega,
    double initial_theta = 0.5,
    double initial_eta = 0.5) {
  const auto time_start = std::chrono::steady_clock::now();

  const int n = X.n_rows;
  const int p = X.n_cols;
  const int q = Y.n_cols;
  const int q_binary = binidxend + 1;

  if (Y.n_rows != static_cast<arma::uword>(n)) {
    Rcpp::stop("X and Y must have the same number of rows.");
  }
  if (weights.n_elem != static_cast<arma::uword>(n) ||
      arma::any(weights <= 0.0) || !weights.is_finite()) {
    Rcpp::stop("weights must contain n finite, strictly positive values.");
  }
  if (q_binary < 0 || q_binary > q) {
    Rcpp::stop("binidxend is inconsistent with ncol(Y).");
  }
  if (center_std.n_rows != static_cast<arma::uword>(p) ||
      center_std.n_cols != static_cast<arma::uword>(q)) {
    Rcpp::stop("center_std must be a p by q matrix.");
  }
  if (initial_B.n_rows != static_cast<arma::uword>(p) ||
      initial_B.n_cols != static_cast<arma::uword>(q)) {
    Rcpp::stop("initial_B must be a p by q matrix.");
  }
  if (initial_Omega.n_rows != static_cast<arma::uword>(q) ||
      initial_Omega.n_cols != static_cast<arma::uword>(q)) {
    Rcpp::stop("initial_Omega must be a q by q matrix.");
  }
  if (lambda1 <= 0.0 || lambda0 < lambda1 ||
      xi1 <= 0.0 || xi0 < xi1) {
    Rcpp::stop("Require lambda0 >= lambda1 > 0 and xi0 >= xi1 > 0.");
  }
  if (theta_hyper_params.n_elem != 2u || eta_hyper_params.n_elem != 2u ||
      arma::any(theta_hyper_params <= 0.0) ||
      arma::any(eta_hyper_params <= 0.0)) {
    Rcpp::stop("The Beta-prior hyperparameter vectors must be positive and length two.");
  }
  if (nrep < 2 || nskp < 1) {
    Rcpp::stop("nrep must be at least 2 and nskp must be positive.");
  }

  const double n_eff = arma::accu(weights);
  if (!std::isfinite(n_eff) || n_eff <= 0.0) {
    Rcpp::stop("The sum of the observation weights must be positive.");
  }

  // Use the same unweighted centering and scaling in every bootstrap draw.
  // This keeps the prior on one common coefficient parameterization.
  arma::vec x_mean = arma::mean(X, 0).t();
  arma::vec x_scale(p, arma::fill::zeros);
  for (int j = 0; j < p; ++j) {
    X.col(j) -= x_mean(j);
    x_scale(j) = arma::norm(X.col(j), 2) / std::sqrt(static_cast<double>(n));
    if (!std::isfinite(x_scale(j)) || x_scale(j) <= BB_EPS) {
      Rcpp::stop("Every column of X must have positive finite variance.");
    }
    X.col(j) /= x_scale(j);
  }

  arma::mat initial_B_std = initial_B;
  initial_B_std.each_col() %= x_scale;
  arma::mat centered_B = initial_B_std - center_std;

  arma::mat Omega = 0.5 * (initial_Omega + initial_Omega.t());
  arma::mat Sigma;
  if (!arma::inv_sympd(Sigma, Omega)) {
    Rcpp::stop("initial_Omega must be symmetric positive definite.");
  }

  double theta = clamp_probability(initial_theta);
  double eta = clamp_probability(initial_eta);

  BBMixedWorkingParam working(X, Y, weights, center_std, q_binary);
  arma::vec mu_old = working.mu;
  working.update(mu_old, centered_B, Sigma, nrep, nskp);
  mu_old = working.mu;

  arma::mat q_star(q, q, arma::fill::zeros);
  arma::mat xi_star(q, q, arma::fill::zeros);

  bool converged = false;
  bool early_terminate = false;
  bool objective_terminate = false;
  int objective_counter = 0;
  int iterations = 0;
  double objective_value = bb_objective(
    n_eff, p, q, working.S, centered_B, Omega,
    lambda1, lambda0, xi1, xi0, theta, eta, diag_penalty,
    theta_hyper_params, eta_hyper_params
  );

  const int quic_max_iter_base = std::max(5 * max_iter, 50);

  for (iterations = 1; iterations <= max_iter; ++iterations) {
    if (iterations % 25 == 0) Rcpp::checkUserInterrupt();

    const arma::mat old_B = centered_B;
    const arma::mat old_Omega = Omega;
    const double old_objective = objective_value;

    for (int k = 0; k < q; ++k) {
      for (int kk = k + 1; kk < q; ++kk) {
        const double probability = slab_probability(
          Omega(k, kk), eta, xi1, xi0
        );
        q_star(k, kk) = probability;
        q_star(kk, k) = probability;
      }
    }
    q_star.diag().zeros();

    xi_star = xi1 * q_star + xi0 * (1.0 - q_star);
    if (diag_penalty == 1) {
      xi_star.diag().fill(xi1);
    } else {
      xi_star.diag().zeros();
    }

    bb_update_B_theta(
      n_eff, p, q, centered_B, working.R, working.tXR, working.S,
      theta, Omega, working.X, working.tXX, lambda1, lambda0,
      theta_hyper_params, max_iter, eps, verbose
    );

    if (q > 1) {
      const double a_eta = eta_hyper_params(0);
      const double b_eta = eta_hyper_params(1);
      const double denominator =
        a_eta + b_eta - 2.0 + static_cast<double>(q * (q - 1)) / 2.0;
      if (denominator > 0.0) {
        eta = clamp_probability(
          (a_eta - 1.0 + arma::accu(q_star) / 2.0) / denominator
        );
      }
    }

    xi_star /= n_eff;
    int quic_max_iter = quic_max_iter_base;
    arma::cube quic_result = my_quic(
      q, working.S, xi_star, eps, quic_max_iter
    );
    Omega = quic_result.slice(0);
    Sigma = quic_result.slice(1);
    rescale_binary_covariance(Sigma, Omega, q_binary);

    working.update(mu_old, centered_B, Sigma, nrep, nskp);
    mu_old = working.mu;

    objective_value = bb_objective(
      n_eff, p, q, working.S, centered_B, Omega,
      lambda1, lambda0, xi1, xi0, theta, eta, diag_penalty,
      theta_hyper_params, eta_hyper_params
    );

    const double objective_gain =
      (objective_value - old_objective) /
      std::max(1.0, std::abs(old_objective));
    if (!std::isfinite(objective_value) || objective_gain < eps) {
      ++objective_counter;
    } else {
      objective_counter = 0;
    }

    const arma::umat B_support_changed =
      arma::conv_to<arma::umat>::from((old_B != 0.0) != (centered_B != 0.0));
    const arma::umat Omega_support_changed =
      arma::conv_to<arma::umat>::from((old_Omega != 0.0) != (Omega != 0.0));
    const double B_change =
      arma::norm(centered_B - old_B, "fro") /
      std::max(1.0, arma::norm(old_B, "fro"));
    const double Omega_change =
      arma::norm(Omega - old_Omega, "fro") /
      std::max(1.0, arma::norm(old_Omega, "fro"));

    converged =
      arma::accu(B_support_changed) == 0u &&
      arma::accu(Omega_support_changed) == 0u &&
      B_change <= eps && Omega_change <= eps;

    if (working.s_eval.n_elem > 0u) {
      const double min_eigen = working.s_eval.min();
      const double max_eigen = working.s_eval.max();
      const double condition =
        min_eigen > BB_EPS ? max_eigen / min_eigen :
          std::numeric_limits<double>::infinity();
      early_terminate = condition > s_max_condition;
    }

    if (verbose == 1) {
      Rcpp::Rcout << "[BB-SSL] iteration " << iterations
                  << ", objective = " << objective_value
                  << ", active B deviations = " << arma::accu(centered_B != 0.0)
                  << ", B change = " << B_change
                  << ", Omega change = " << Omega_change << "\n";
    }

    if (objective_counter >= obj_counter_max) {
      objective_terminate = true;
      break;
    }
    if (early_terminate || converged) break;
  }

  if (iterations > max_iter) iterations = max_iter;

  arma::mat B_std = centered_B + center_std;
  arma::mat B = B_std;
  B.each_col() /= x_scale;

  arma::mat centered_B_original = centered_B;
  centered_B_original.each_col() /= x_scale;
  arma::mat center_original = center_std;
  center_original.each_col() /= x_scale;

  arma::vec alpha = working.mu - B.t() * x_mean;

  const auto time_end = std::chrono::steady_clock::now();
  const double elapsed_seconds =
    std::chrono::duration_cast<std::chrono::duration<double>>(
      time_end - time_start
    ).count();

  return Rcpp::List::create(
    Rcpp::Named("alpha") = alpha,
    Rcpp::Named("B") = B,
    Rcpp::Named("Omega") = Omega,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("centered_B") = centered_B_original,
    Rcpp::Named("center") = center_original,
    Rcpp::Named("centered_B_std") = centered_B,
    Rcpp::Named("center_std") = center_std,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("eta") = eta,
    Rcpp::Named("objective") = objective_value,
    Rcpp::Named("iterations") = iterations,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("early_terminate") = early_terminate,
    Rcpp::Named("objective_terminate") = objective_terminate,
    Rcpp::Named("elapsed_seconds") = elapsed_seconds,
    Rcpp::Named("x_center") = x_mean,
    Rcpp::Named("x_scale") = x_scale,
    Rcpp::Named("n_eff") = n_eff
  );
}
