function expectation = argmin_sum(prob, num_heads)

num_test = length(prob);
cumsum_p = cumsum(prob);
first_idx = find(cumsum_p>=num_heads, 1);
if isempty(first_idx)
  expectation = num_test;
else
  extra = cumsum_p(first_idx) - num_heads;
  expectation = (first_idx - extra / prob(first_idx));
end
