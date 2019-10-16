#include <random>
#include <iostream>

double* pmf_of_poisson_binom(int n, double *probs, int num_heads)
{
  /* compute probability table of Poisson binomial distribution */
  
  int N_ROW = n+1;  /* +1 to include the case of (0, 0) */
  int N_COL = num_heads+1;
  int i, j;
  /*
   * define p_table such that p_table(i*N_COLj) means the probability
   * of j heads when tossing i coins
   */
  double *p_table = new double[N_ROW * N_COL];
  
  double log_cumprod = log(1 - probs[0]);
  
  p_table[0] = 1;  /* toss one coin, get one head with probability 1 */
  
  for (i = 1; i < N_ROW; i++){
    /*
     * toss i coins with 0 head,
     * probability is product of (1-p_k), k=1..i
     */
    p_table[i*N_COL] = exp(log_cumprod);
    log_cumprod += log(1 - probs[i]);
  }
  
  for (j = 1; j < N_COL; j++){
    for (i = 0; i < j; i++){
      /*
       * when tossing i coins,
       * it's not possible to get more than i heads, hence zero prob
       */
      p_table[i*N_COL+j] = 0;
    }
    for (i = j; i < N_ROW; i++){
      /*
       * toss i coins, get j heads (j >= i)
       * probability can be computed as follows:
       * Pr(the i-th coin is head) * Pr(first i-1 coins with j-1 heads)
       * + Pr(the i-th coin is tail) * Pr(first i-1 coins with j heads)
       */
      p_table[i*N_COL+j] = probs[i-1] * p_table[(i-1)*N_COL+j-1] +
              (1 - probs[i-1]) * p_table[(i-1)*N_COL+j];
    }
  }
  return p_table;
}
struct Neg_poisson_binomial_dist{
  int n = 0;
  int num_heads = 0;
  double *heads_probs = NULL;
  double *pmf = NULL;
  double expectation = 0;
  double approx_one = 1;
  int max_increment = n;
};

Neg_poisson_binomial_dist approx_pmf_of_neg_poisson_binom(int n, 
        double *probs, int num_heads, double approx_one)
{
  /* compute approximate negative Poisson binomial distribution */

  /* define arrays to store the dynamic programming table 
   * alternate to save memory
   */
  double *p_old = new double[num_heads+1];
  double *p_new = new double[num_heads+1];
  double *tmp;  /* for swapping */
  
  double *pm = new double[n-num_heads+1];
  int n_heads, n_coins, increment;
  double expectation = 0;
  
  double npb_sum = 0;  /* cumulative sum of probabilities */

  p_old[0] = 1;  /* toss one coin, get one head with probability 1 */

  for (n_heads = 1; n_heads <= num_heads; n_heads++){
    /*
     * toss i coins with i heads,
     * probability is product of p_k, k=1..i
     */
    p_old[n_heads] = p_old[n_heads-1] * probs[n_heads-1];
  }
  
  pm[0] = p_old[num_heads];
  expectation += pm[0] * num_heads;
  npb_sum += pm[0];
  
  for (increment = 1; increment <= n - num_heads; increment++){
    
    p_new[0] = p_old[0] * (1 - probs[increment-1]);
    
    for (n_heads = 1; n_heads < num_heads; n_heads++){
      /* 
       * If Pr(n, r) denote the probability of n coins with r heads, then
       * p_new[n_heads] is Pr(n_heads+increment,   n_heads)
       * p_old[n_heads] is Pr(n_heads+increment-1, n_heads)
       */
      n_coins = n_heads + increment;
      p_new[n_heads] = probs[n_coins-1] * p_new[n_heads-1] + 
              (1-probs[n_coins-1]) * p_old[n_heads];
    }
    
    pm[increment] = probs[num_heads+increment-1] * p_new[num_heads-1];
    expectation += pm[increment] * (num_heads + increment);
    
    tmp = p_old;
    p_old = p_new;
    p_new = tmp;
    
    npb_sum += pm[increment];
    if (npb_sum > approx_one)
      break;
  }
  Neg_poisson_binomial_dist res;
  res.n = n;
  res.num_heads = num_heads;
  res.heads_probs = probs;
  res.pmf = pm;
  res.expectation = expectation;
  res.approx_one = approx_one;
  res.max_increment = increment;
  
  delete p_old;
  delete p_new;
  delete pm;
  
  return res;
}

double approx_exp_of_neg_poisson_binom(int n, 
        double *probs, int num_heads, double approx_one)
{
  /* compute approximate negative Poisson binomial distribution */

  /* define arrays to store the dynamic programming table 
   * alternate to save memory
   */
  double *p_old = new double[num_heads+1];
  double *p_new = new double[num_heads+1];
  double *tmp;  /* for swapping */
  
  double pm;
  int n_heads, n_coins, increment;
  double expectation = 0;
  
  double npb_sum = 0;  /* cumulative sum of probabilities */

  p_old[0] = 1;  /* toss one coin, get one head with probability 1 */

  for (n_heads = 1; n_heads <= num_heads; n_heads++){
    /*
     * toss i coins with i heads,
     * probability is product of p_k, k=1..i
     */
    p_old[n_heads] = p_old[n_heads-1] * probs[n_heads-1];
  }
  
  expectation += p_old[num_heads] * num_heads;
  npb_sum += p_old[num_heads];
  
  for (increment = 1; increment <= n - num_heads; increment++){
    
    p_new[0] = p_old[0] * (1 - probs[increment-1]);
    
    for (n_heads = 1; n_heads < num_heads; n_heads++){
      /* 
       * If Pr(n, r) denote the probability of n coins with r heads, then
       * p_new[n_heads] is Pr(n_heads+increment,   n_heads)
       * p_old[n_heads] is Pr(n_heads+increment-1, n_heads)
       */
      n_coins = n_heads + increment;
      p_new[n_heads] = probs[n_coins-1] * p_new[n_heads-1] + 
              (1-probs[n_coins-1]) * p_old[n_heads];
    }
    
    pm = probs[num_heads+increment-1] * p_new[num_heads-1];
    expectation += pm * (num_heads + increment);
    
    tmp = p_old;
    p_old = p_new;
    p_new = tmp;
    
    npb_sum += pm;
    if (npb_sum > approx_one)
      break;
  }

  delete p_old;
  delete p_new;
  
  return expectation;
}

double* pmf_of_neg_poisson_binom(int n, double *probs, int num_heads)
{
  /* compute negative Poisson binomial distribution */
  
  double *p_table = pmf_of_poisson_binom(n, probs, num_heads);
  double *pgnb = new double[n-num_heads+1];
  int N_COL = num_heads+1;
  for (int i = num_heads; i <= n; i++){
    /*
     * to get "num_heads" heads
     * the probability of tossing i coins is (i>=num_heads):
     * (note the i-th coin must be head)
     * Pr(the i-th coin is head) * Pr(the first i-1 coins with
     * "num_heads-1" heads)
     */
    pgnb[i-num_heads] = probs[i-1]*p_table[(i-1)*N_COL+num_heads-1];
  }
  return pgnb;
}

double* generate_uniform_random_vector(int n, int seed){
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  double *probs = new double[n];
  for (int i = 0; i < n; i++){
    probs[i] = uniform(generator);
  }
  return probs;
}

int* sample_from_neg_poisson_binom(int n, double *probs,
        int num_heads, int num_samples, int seed){
  /*
   * return samples from a negative Poisson binomial distribution (NPBD)
   * Let there be n coins with different probabilities (probs) of head,
   * given number of heads (num_heads) required, how many coins
   * do we need to toss to have this many heads?
   * We call the number of coins tossed a negative Poisson binomial variable
   *
   * Args:
   *  n: number of coins
   *  probs: probabilities of the coins
   *  num_heads: number of heads required
   *  num_samples: number of Monte Carlo samples
   *
   * Returns:
   *  samples: samples of the given NPBD
   */
  
  int count_heads, i, j;
  double rand_num = 0.0;
  int *samples = new int[num_samples];
  
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  
  for (i = 0; i < num_samples; i++){
    count_heads = 0;
    for (j = 0; j < n; j++){
      rand_num = uniform(generator);
      if (rand_num <= probs[j]){
        /* head */
        count_heads ++;
        if (count_heads == num_heads){
          samples[i] = j + 1;
          break;
        }
      }
    }
  }
  return samples;
}

double expectation_of_neg_poisson_binom(int n, double *probs, int num_heads, 
        int method, int num_samples, int seed)
{
  /* compute expectation of negative Poisson binomial distribution */
  
  double expectation = 0.0; 
  if (method == 1){
    double *pgnb = pmf_of_neg_poisson_binom(n, probs, num_heads);
    for (int i = num_heads; i <= n; i++){
      expectation += pgnb[i-num_heads] * double(i);
    }
  }
  else if (method == 2){
    int *samples = sample_from_neg_poisson_binom(n, probs, num_heads, 
            num_samples, seed);
    for (int i = 0; i < num_samples; i++){
      expectation += (double)samples[i];
    }
    expectation /= (double)num_samples;
  }
  return expectation;
}

