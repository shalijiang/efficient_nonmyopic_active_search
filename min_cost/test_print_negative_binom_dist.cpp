#include <cstdio>
#include <cstdlib>
#include "negative_binomial_distribution.h"


int main(int argc, char* argv[]){
  int i, j;
  int n = 100, num_heads = 5, num_samples = 10, seed = 3;
  double *pmf;
  Neg_poisson_binomial_dist dist;
//   printf("argc: %d\n", argc);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) num_heads = atoi(argv[3]);
  if (argc > 4) num_samples = atoi(argv[4]);
  if (argc > 5) seed = atoi(argv[5]);
  double* probs = generate_uniform_random_vector(n, seed);
//   for (i = 0; i < 10; i ++) printf("%f ", probs[i]);
//   printf("\n");
  int which = atoi(argv[1]);
  if (which == 1){
    pmf = pmf_of_neg_poisson_binom(n, probs, num_heads);
    for (i = num_heads; i <= n; i++){
      if (pmf[i-num_heads] > 1e-6/n)
        printf("%d %f\n", i, pmf[i-num_heads]);
    }
  }
  else if (which == 2){
    double approx_one = 1-1e-9;
    dist = approx_pmf_of_neg_poisson_binom(n, probs, num_heads, approx_one);
    pmf = dist.pmf;
    for (i = 0; i <= dist.max_increment; i++){
      printf("%d %f\n", i+num_heads, pmf[i]);
    }
  }
  else if (which == 3){
    int* samples = sample_from_neg_poisson_binom(n, probs, num_heads, 
            num_samples, seed);
    for (i = 0; i < num_samples; i++){
      printf("%d\n", samples[i]);
    }
  }
  return 0;
}