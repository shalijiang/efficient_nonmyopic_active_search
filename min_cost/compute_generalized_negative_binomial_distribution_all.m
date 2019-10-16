function [pgnb, pgb] = ...
  compute_generalized_negative_binomial_distribution_all(p, num_r)
% Given n coins with HEAD probabilities sorted as p1, p2, ..., pn, 
% flip these coins one by one from 1 to n.
% Let m be coins fliped to when there are r HEADs.
% This function compute the probability mass function of m. 
%
% In the computation of generalized negative binomial distribution (pgnb),
% we will use generalized binomial distribution (pgb). That is,
% flip n coins with HEAD probabilities p1, p2, ..., pn, 
% the number of HEADs follows generalized binomial distribution.
% 
% pgb can be computed via the following recursion
%   pgb(n, r=0) = prod(1-p)
%   pgb(n, 0<r<=n) = pn * pgb(n-1, r-1) + (1-pn) * pgb(n-1, r)
%   pgb(n, r>n) = 0
%
% now pgnb can be computed using pgb as follows:
%   pgnb(m<r,  r) = 0  (you must flip at least r coins to get r HEADs)
%   pgnb(m>=r, r) = pm * pgb(m-1, r-1)
% 
% Input:
%   p: probabilities p1, p2, ..., pn
%   num_r: number of HEADs required
% 
% Output:
%   pgnb: a matrix of size (n+1, num_r+1), 
%         where pngb(i, r) is the probability of i-1 flip when r-1 HEADs 
%         are required
%   pgb: also a matrix of size (n+1, num_r+1), representing the probability
%        mass function of the generalized binomial distribution:
%        pgb(i, r) is the probability r HEAD when fliping the first i coins
%
% 2018/11/6
n = length(p);
pgb = zeros(n+1, num_r+1);
pgb(1, 1) = 1;
pgb(2:end, 1) = cumprod(1 - p)';
for r = 2:num_r+1
  for i = r:n+1
    pgb(i, r) = p(i-1) * pgb(i-1, r-1) + (1-p(i-1))*pgb(i-1, r);
  end
end

pgnb = zeros(n+1, num_r+1);
pgnb(1,1) = 1;
for r = 2:num_r+1
  for i = r:n+1
    pgnb(i, r) = p(i-1) * pgb(i-1, r-1);
  end
end