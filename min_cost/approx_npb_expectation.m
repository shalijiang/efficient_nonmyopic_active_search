function approx_npb_expectation(p, r)

n = length(p);
i = 1;
res = 0;
while r > 0
  e = r / p(i);
  j = i+1;
  while p(j) == p(i)
    j = j + 1;
  end
  l = j - i;
  res = res + l*p(i);
  r = r - l*p(i);
end