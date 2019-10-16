#include "mex.h"
#include "math.h"

#define P_ARG       prhs[0]
#define Q_ARG       prhs[1]
#define TOP_IND_ARG prhs[2]
#define REM_GOAL    prhs[3]
#define EXP_ARG     plhs[0]

#define P_IND ((int)(top_ind[j]) - 1)

void mexFunction(int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
  
  double *p, *q, *top_ind;
  int remaining_goal, i, j, k, ii;
  size_t n, m;

  /* get input */
  p = mxGetPr(P_ARG);
  q = mxGetPr(Q_ARG);
  top_ind = mxGetPr(TOP_IND_ARG);
  remaining_goal = (int)(mxGetScalar(REM_GOAL));

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
  int N_ROW = m;
  int N_COL = remaining_goal+1;
  double *pgb = new double[N_ROW * N_COL];
  double log_cumprod = log(1 - pp[0]);
  pgb[0] = 1;
  for (i = 1; i < N_ROW; i++){
    pgb[i*N_COL] = exp(log_cumprod);
    log_cumprod += log(1 - pp[i]); 
  }
  for (j = 1; j < N_COL; j++){
    for (i = 0; i < j; i++){
      pgb[i*N_COL+j] = 0;
    }
    for (i = j; i < N_ROW; i++){
      pgb[i*N_COL+j] = pp[i-1]*pgb[(i-1)*N_COL+j-1] + 
              (1-pp[i-1])*pgb[(i-1)*N_COL+j];
    }
  }
  
  double pgnb;
  double expectation = 0;
  for (i = remaining_goal; i < N_ROW; i++){
    pgnb = pp[i-1]*pgb[(i-1)*N_COL+remaining_goal-1];
    expectation += pgnb * i;
  }
  delete [] pp;
  delete [] pgb;
  EXP_ARG = mxCreateDoubleScalar(expectation);
}