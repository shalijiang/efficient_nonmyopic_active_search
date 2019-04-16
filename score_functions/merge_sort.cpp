#include "mex.h"

#define P_ARG       prhs[0]
#define Q_ARG       prhs[1]
#define TOP_IND_ARG prhs[2]
#define BUDGET_ARG  prhs[3]
#define SUM_ARG     plhs[0]
#define ALL_ARG     plhs[1]

#define P_IND ((int)(top_ind[j]) - 1)

void mexFunction(int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {

  double *p, *q, *top_ind, sum;
  int budget, i, j, k;
  size_t n;

  /* get input */
  p = mxGetPr(P_ARG);
  q = mxGetPr(Q_ARG);
  top_ind = mxGetPr(TOP_IND_ARG);
  budget = (int)(mxGetScalar(BUDGET_ARG));

  n = mxGetNumberOfElements(Q_ARG);

  sum = 0;
  i = 0;
  j = 0;
  while (p[P_IND] == 0)
    j++;
  k = 0;

  while ((i < budget) && (k < n)) {
    if (p[P_IND] > q[k]) {
      sum += p[P_IND];
      do {
	    j++;
      } while (p[P_IND] == 0);
    }
    else {
      sum += q[k];
      k++;
    }

    i++;
  }

  while (i < budget) {
    sum += p[P_IND];
    do {
      j++;
    } while (p[P_IND] == 0);

    i++;
  }

  SUM_ARG = mxCreateDoubleScalar(sum);

}
