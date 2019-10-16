function m_expectation_monte_carlo = ...
  compute_generalized_negative_binomial_expectation(p, num_r, ...
  num_sample_paths)
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
%   the expectation of m
%
% 2018/11/8
if nargin < 3
  num_sample_paths = 32768;
end
n = length(p);
m_samples = nan(num_sample_paths, 1);
unif = rand(num_sample_paths, n);
coin_heads = unif < p;
num_heads = cumsum(coin_heads, 2);

for i = 1:num_sample_paths
  m_samples(i) = find(num_heads(i,:) >= num_r, 1);
end

m_expectation_monte_carlo = mean(m_samples);