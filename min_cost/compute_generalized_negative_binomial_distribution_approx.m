function expectation = ...
  compute_generalized_negative_binomial_distribution_approx(p, num_r, approx_one)
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
old = zeros(1, num_r+1);
new = zeros(1, num_r+1);

prob_sum = 0;
expectation = 0;

old(1) = 1;
old(2:end) = cumprod(p(1:num_r));
expectation = expectation + old(end) * num_r;
prob_sum = prob_sum + old(end);

for increment = 1:n-num_r
  new(1) = old(1) * (1 - p(increment));
  for n_heads = 1:num_r
    n_coins = n_heads + increment;
    new(n_heads+1) = p(n_coins) * new(n_heads) + ...
      (1-p(n_coins)) * old(n_heads+1);
  end
  pm = p(num_r+increment) * new(num_r);
  expectation = expectation + pm * (num_r + increment);
  tmp = old;
  old = new;
  new = tmp;
  
  prob_sum = prob_sum + pm;
  if prob_sum > approx_one
%     fprintf('stoped at %d coins\n', num_r + increment);
    break
  end
end
