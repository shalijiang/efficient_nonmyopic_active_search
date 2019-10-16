#include "mex.h"
#include "math.h"
#include <random>
#include "negative_binomial_distribution.h"

#define P_ARG       prhs[0]
#define REM_GOAL    prhs[1]
#define APPROX_ONE  prhs[2]

#define EXP_ARG     plhs[0]

void mexFunction(int nlhs,       mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
  
  double *p, approx_one, expectation;
  int remaining_goal, m;

  /* get input */
  p = mxGetPr(P_ARG);
  remaining_goal = (int)(mxGetScalar(REM_GOAL));
  approx_one = (double)(mxGetScalar(APPROX_ONE));  // 1 for dp, 2 for Monte Carlo
  m = mxGetNumberOfElements(P_ARG);

  expectation = approx_exp_of_neg_poisson_binom(m-1, p, remaining_goal, 
          approx_one);

  EXP_ARG = mxCreateDoubleScalar(expectation);
}