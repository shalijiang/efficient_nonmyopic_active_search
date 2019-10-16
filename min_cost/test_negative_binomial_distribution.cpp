#include <cstdio>
#include <random>
#include <cstdlib>
#include <algorithm>
#include "negative_binomial_distribution.h"

using namespace std;

const int DP = 1;
const int MONTE_CARLO = 2;

int test_expectation(int n, double* probs, int num_heads, 
        int N, int num_repeats){
  
  int i, j;
  double expectation1, expectation11, expectation12, expectation2, error, 
          approx_one = 1-1e-9;
  double clock_time_dp, clock_time_approx_dp, clock_time_approx_dp1;
  Neg_poisson_binomial_dist dist;
  clock_t start = clock();
  expectation1 = expectation_of_neg_poisson_binom(n, probs, num_heads,
          DP, 0, 0);
  clock_t end = clock();
  clock_time_dp = (double) (end-start) / CLOCKS_PER_SEC;
  printf("dp expectation=%f\n", expectation1);
  
  start = clock();
  dist = approx_pmf_of_neg_poisson_binom(n, probs, num_heads, approx_one);
  end = clock();
  clock_time_approx_dp = (double) (end-start) / CLOCKS_PER_SEC;
  expectation11 = dist.expectation;
  printf("approx dp expectation=%f error=%f\n", 
          expectation11, fabs(expectation1 - expectation11));
  
  start = clock();
  expectation12 = approx_exp_of_neg_poisson_binom(n, probs, num_heads, approx_one);
  end = clock();
  clock_time_approx_dp1 = (double) (end-start) / CLOCKS_PER_SEC;
  printf("approx dp1 expectation=%f error=%f\n", 
          expectation12, fabs(expectation1 - expectation12));
  
  double *errors = new double[num_repeats];
  
  int start_n = 1<<9;
  int num_samples = start_n;
  double *ave_errors = new double[N], ave_error;
  double *clock_time_monte = new double[N], clock_time;
  for (i = 0; i < N; i++){
    num_samples <<= 1;
    clock_time = 0.0;
    for (j = 0; j < num_repeats; j ++){
      start = clock();
      expectation2 = expectation_of_neg_poisson_binom(n, probs, num_heads,
              MONTE_CARLO, num_samples, j+1);
      end = clock();
      clock_time += (double) (end-start) / CLOCKS_PER_SEC;
      error = fabs(expectation1-expectation2);
//       printf("monte carlo (%d samples) expectation=%f, error=%f\n",
//               num_samples, expectation2, error);
      errors[j] = error;
    }
    clock_time_monte[i] = clock_time/num_repeats;
    ave_error = 0.0;
    for (j = 0; j < num_repeats; j ++){
      ave_error += errors[j];
    }
    ave_error /= (double)num_repeats;
    ave_errors[i] = ave_error;
    
  }
  num_samples = start_n;
  printf("DP time: %f\nApprox DP time: %f\nApprox DP exp time: %f\n", 
          clock_time_dp, 
          clock_time_approx_dp,
          clock_time_approx_dp1);
  
  for (i = 0; i < N; i++){
    num_samples <<= 1;
    int greater = clock_time_monte[i] > clock_time_dp;
    printf("%10d samples: average error: %f time: %f (longer than DP? %d)\n", 
            num_samples, ave_errors[i], clock_time_monte[i], greater);
  }
  return 0;
}

int main(int argc, char* argv[]){
  int i, j;
  int n = 100000, num_heads = 100, seed = 3, N = 3, num_repeats = 1, num_seeds=1;
  
  printf("Testing expecatation ...\n");
  printf("n=%d, num_heads=%d\n", n, num_heads);
  for (seed = 1; seed <= num_seeds; seed++){
    printf("Seed %d ...\n", seed);
    double *probs = generate_uniform_random_vector(n, seed);
    for (i = 0; i < std::min(10, n); i++){
      printf("%.2f ", probs[i]);
    }
    printf("\n");
    test_expectation(n, probs, num_heads, N, num_repeats);
  }
  return 0;
}