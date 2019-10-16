%% test correctness c++ implementation of the negative binomial distribution
%% Plot the probability mass function computed by DP against that by
%% Monte Carlo sampling
close all
files = {'pmf.log', 'pmf1.log', 'samples.log'};
n = 100000;
num_heads = 500;
num_samples = 100000;
seed = 5;
for method = 1:3
  cmd = sprintf('./print %d %d %d %d %d > %s', ...
    method, n, num_heads, num_samples, seed, files{method});
  tic
  system(cmd)
  toc
  load(files{method});
end
m = length(pmf);
N = length(samples);
figure; hold on
idx1 = find(pmf(:,2) > 1e-4);
idx1 = min(idx1):max(idx1);
num_heads = pmf(1,1);
plot(pmf(idx1,1), pmf(idx1,2));
plot(pmf1(:,1), pmf1(:,2), '--');

counts = tabulate(samples);
probs = counts(:,2)/N;
plot(counts(pmf(idx1,1),1), probs(pmf(idx1,1)), ':');
legend('DP', 'approx DP', 'Monte Carlo');
xlabel('number of coins tosses');
ylabel('probability');