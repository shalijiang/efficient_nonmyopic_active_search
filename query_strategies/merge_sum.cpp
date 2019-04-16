#include "mex.h"

#define P_ARG       prhs[0]
#define Q_ARG       prhs[1]
#define TOP_IND_ARG prhs[2]
#define REM_GOAL    prhs[3]



#define SUM_ARG     plhs[0]
#define ALL_ARG     plhs[1]

#define P_IND ((int)(top_ind[j]) - 1)

void mexFunction(int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
  
  double *p, *q, *top_ind, utility;
  int remaining_goal, i, j, k;
  double cost;
  size_t n, m;

  /* get input */
  p = mxGetPr(P_ARG);
  q = mxGetPr(Q_ARG);
  top_ind = mxGetPr(TOP_IND_ARG);
  remaining_goal = (int)(mxGetScalar(REM_GOAL));
  

  n = mxGetNumberOfElements(Q_ARG);
  m = mxGetNumberOfElements(P_ARG);
  
  utility = 0;
  cost = 0;
  i = 0;
  j = 0;
  while (p[P_IND] == 0)
    j++;
  k = 0;
  double last_p;
  while ((j < m) && (k < n)) {
    if (p[P_IND] > q[k]) {
      utility += p[P_IND];
      last_p = p[P_IND];
      do {
        j++;
      } while (p[P_IND] == 0);
    }
    else {
      utility += q[k];
      last_p = q[k];
      k++;
    }
    cost += 1;
    if (utility >= remaining_goal){
      break;
    }
  }
  if (utility < remaining_goal){
    while (j < m) {
      utility += p[P_IND];
      last_p = p[P_IND];
      do {
        j++;
      } while (p[P_IND] == 0);
      
      cost += 1;
      if (utility >= remaining_goal){
        break;
      }
    }
    
    while (k < n) {
      utility += q[k];
      last_p = q[k];
      k++;
      cost += 1;
      if (utility >= remaining_goal){
        break;
      }
    }
  }
  if (utility < remaining_goal){
    cost = m-1;
  }
  else{  // corretion by subtracting the extra utility
    cost = cost - (utility - remaining_goal)/last_p;
    //mexPrintf("cost: %f\n", cost);
  }
  SUM_ARG = mxCreateDoubleScalar(cost);
}
