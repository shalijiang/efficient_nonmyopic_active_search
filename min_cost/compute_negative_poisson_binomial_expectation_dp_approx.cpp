#include "mex.h"
#include "math.h"
#include <random>
#include "negative_binomial_distribution.h"

#define P_ARG       prhs[0]
#define Q_ARG       prhs[1]
#define TOP_IND_ARG prhs[2]
#define REM_GOAL    prhs[3]
#define APPROX_ONE  prhs[4]

#define EXP_ARG     plhs[0]


#define P_IND ((int)(top_ind[j]) - 1)

void mexFunction(int nlhs,       mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
  
  double *p, *q, *top_ind, approx_one, expectation;
  int remaining_goal, i, j, k, ii;
  size_t n, m;
  
  /* get input */
  p = mxGetPr(P_ARG);
  q = mxGetPr(Q_ARG);
  top_ind = mxGetPr(TOP_IND_ARG);
  remaining_goal = (int)(mxGetScalar(REM_GOAL));
  approx_one = (double)(mxGetScalar(APPROX_ONE));  // 1 for dp, 2 for Monte Carlo
  
  n = mxGetNumberOfElements(Q_ARG);
  m = mxGetNumberOfElements(P_ARG);
  
  double *pp = new double[m-1];
  i = 0;
  j = 0;
  ii = 0;
  k = 0;
  
  while ((j < m) && (k < n)) {
    while (j < m && p[P_IND] == 0)
      j++;
    if (j >= m) break;
    if (p[P_IND] > q[k]) {
      pp[ii] = p[P_IND];
      j++;
    }
    else {
      pp[ii] = q[k];
      k++;
    }
    ii++;
  }
  
  while (j < m) {
    while (j < m && p[P_IND] == 0)
      j++;
    if (j >= m) break;
    pp[ii++] = p[P_IND];
    j++;
  }
  
  while (k < n) {
    pp[ii++] = q[k++];
  }
  
  if (ii != m-1) {
    mexPrintf("ii %d\n", ii);
  }
  
  expectation = approx_exp_of_neg_poisson_binom(m-1, pp, remaining_goal, 
          approx_one);
  delete pp;
  EXP_ARG = mxCreateDoubleScalar(expectation);
}