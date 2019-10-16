%% Generalized negative binomial distribution
% *Definition* [generalized negative binomial distribution]: Given $n$ coins
% with HEAD probabilities $$p_1, p_2, \dots, p_n$$, flip the coins one by one
% in this order. Let m be coins fliped when there are r HEADs. We say $m$ follows
% a generalized negative binomial distribution: $m \sim GNB(r, [p_1, p_2, \dots,
% p_n])$.
%
% What is the distribution of $m$, and what is the expectation of m? In this
% note, we simulate this process, and present a dynamic programming solution for
% the probability mass function. We also proposed three approximations and show
% how close they are from the true distribution.
close all
rng(20181106);
savefig = true;
n = 30000;
num_r = 500;
pfrom = 'real';

if strcmp(pfrom, 'beta')
  % generate the probabilities
  p = ones(1, n) * .01;
  p(1:n) = betarnd(5, 10, 1, n);
  % p(num_r*2+1:end) = rand(1, n-2*num_r) * .1;
elseif strcmp(pfrom, 'real')
  % probabilities from real data
  num_pos = 100;
  num_neg = 100;
  addpath(genpath('../../active_search'));  % change this accordingly
  addpath(genpath('../'));
  data_name = 'citeseer_data';
  [problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
    load_data(data_name, '../data/');
  model = get_model(@knn_model, weights, alpha);
  model = get_model(@model_memory_wrapper, model);
  positive_ind = find(labels == 1);
  negative_ind = find(labels == 2);
  train_ind = randsample(positive_ind, num_pos);
  train_ind = [train_ind; randsample(negative_ind, num_neg)];
  test_ind = unlabeled_random_selector1(problem, train_ind, Inf);
  probabilities = knn_model(problem, train_ind, labels(train_ind), ...
    test_ind, weights, alpha);
  p = probabilities(:,1)';
end

p = sort(p, 'descend');

%%
% *Definition* [Generalized binomial distribution] Let there be $n$ different
% coins with HEAD probability $$p_1, p_2, \dots, p_n$$. We independently toss
% each coin once; we say the number of HEADs $r$ follows a generalized binomial
% distribution $GB(n, \{p_1, p_2, \cdots, p_n\})$.
%
% The probability mass function (PMF) of $r \sim GB(n, \{p_1, p_2, \cdots,
% p_n\})$ can be computed via dynamic programming:
%
% $$ {\textstyle	\Pr_{GB}(n, r) } = 	\begin{cases}		\prod_{i=1}^{n} (1- p_i),
% & \text{if } r = 0; \\		p_n \Pr_{GB}(n-1, r-1)  + (1 - p_n) \Pr_{GB}(n-1, r),
% & \text{if } 0 < r <= n. \\		0	& \text{if } r > n; \\	\end{cases}$$
%
%
%
% Based on the PMF of GB, we can derive the PMF of GNB:
%
% $$	{\textstyle \Pr_{GNB}(m, r)}	= 	\begin{cases}		0, 	& \text{if } m <
% r; \\		p_m \Pr_{GB}(m-1, r-1),  & \text{if } m \ge r.	\end{cases}$$

% exactly compute the expectation
m_expectation_true   = nan(1, num_r);
m_expectation_eps_dp = nan(1, num_r);
m_expectation_eps_dp1 = nan(1, num_r);
approx_one = 1-1e-6;
[pgnb, pgb] = compute_generalized_negative_binomial_distribution_all(p, num_r);

for r = 2:num_r+1
  m_expectation_true(r-1) = sum(pgnb(r:n+1, r) .* (r-1:n)');
end
%%
for r = 1:num_r
  m_expectation_eps_dp(r) = compute_generalized_negative_binomial_distribution_approx(p, r, approx_one);
  %   m_expectation_eps_dp1(r) = compute_negative_poisson_binomial_expectation_dp_approx_direct(p, r, approx_one);
end
%%
% Plot the GNB PMF: several examples

plot_r = [0, 50, 200, 400];
fig = figure;
mm = 2;
nn = 2;
subplot(mm, nn, 1)
lw = 1;
plot(p(1:num_r*3), 'r', 'linewidth', lw)
for i = 2:length(plot_r)
  subplot(mm, nn, i)
  % figure;
  max_prob = max(pgnb(1:n,plot_r(i)));
  plot_idx = find(pgnb(1:n, plot_r(i)) > max_prob*.01);
  plot(plot_idx, pgnb(plot_idx,plot_r(i)), 'linewidth', lw)
end

% plot_r = [0, 10, 20, 50, 200, 400];
% plot individual and save
fig = figure;
plot(p(1:num_r*3), 'r', 'linewidth', lw)
ylabel('p')
xlabel('points')
box off
if savefig
  figures_directory = '../neurips2019/figures/';
  if ~isdir(figures_directory), mkdir(figures_directory); end
  figure_name = 'probabilities';
  matlab2tikz(sprintf('%s/%s.tex', figures_directory, figure_name), ...
    'height',       '\sbsfigurewidth', ...
    'width',        '\sbsfigurewidth', ...
    'parseStrings', false, ...
    'showInfo',     false, ...
    'extraCode',    sprintf('\\tikzsetnextfilename{%s}', figure_name));
end

for i = 2:length(plot_r)
  % figure;
  max_prob = max(pgnb(1:n,plot_r(i)));
  plot_idx = find(pgnb(1:n, plot_r(i)) > .000001);
  figure; 
  plot(plot_idx, pgnb(plot_idx,plot_r(i)), 'linewidth', lw)
  xlabel('m');
  ylabel('p(m)');
  box off
  if savefig
    figures_directory = '../neurips2019/figures/';
    if ~isdir(figures_directory), mkdir(figures_directory); end
    figure_name = sprintf('npb_distribution_r_%d', plot_r(i));
    matlab2tikz(sprintf('%s/%s.tex', figures_directory, figure_name), ...
      'height',       '\sbsfigurewidth', ...
      'width',        '\sbsfigurewidth', ...
      'parseStrings', false, ...
      'showInfo',     false, ...
      'extraCode',    sprintf('\\tikzsetnextfilename{%s}', figure_name));
  end
end

%%
% Another two heuristic approximations

%% approximation 1
% 1/p1 + 1/p2 + ... + 1/pr
m_expectation_approx1 = cumsum(1./p(1:num_r));

%% approximation 2
m_expectation_approx2 = nan(1, num_r);
m_expectation_approx3 = nan(1, num_r);
cumsum_p = cumsum(p);
for r = 1:num_r
  n_flips = find(cumsum_p >= r, 1);
  try
    m_expectation_approx3(r) = n_flips;
  catch
    continue;
  end
  extra = (cumsum_p(n_flips) - r);
  n_flips = n_flips - extra / p(n_flips);
  m_expectation_approx2(r) = n_flips;
end
%%
% See how close these approximations are in terms of the expecation

approx = [m_expectation_true; ...
  m_expectation_eps_dp; ...
  m_expectation_approx2; ...
  m_expectation_approx1];
style = {'-k', '-.b', '--r', ':g'};

figure;
hold on
ii=num_r;
for i = 1:size(approx, 1)
  plot(1:ii, approx(i,1:ii), style{i});
end
xlabel('r')
ylabel('E[m]')
legend('true','$\\epsilon$-\\DP',  'argmin sum', 'sum 1/p',  ...
  'location', 'northwest', 'Interpreter','latex')
legend boxoff
if savefig
  figures_directory = '../neurips2019/figures/';
  if ~isdir(figures_directory), mkdir(figures_directory); end
  figure_name = 'negative_poisson_binomial_approx_r200';
  matlab2tikz(sprintf('%s/%s.tex', figures_directory, figure_name), ...
    'height',       '\figureheight', ...
    'width',        '\figurewidth', ...
    'parseStrings', false, ...
    'showInfo',     false, ...
    'extraCode',    sprintf('\\tikzsetnextfilename{%s}', figure_name));
end

difference1 = ...
  [m_expectation_eps_dp; ...
  m_expectation_approx3;
  m_expectation_approx2; ...
  m_expectation_approx1] - m_expectation_true;
rmse1 = sqrt(mean(difference1.^2, 2));
disp(rmse1);
figure; hold on
style = {'-', '.', '-', '-'};
idx = [5, 6, 1, 7];
colors = [31, 120, 180; 51, 160, 44; 227, 26, 28; 166, 206, 227]/255;
sample_colors = [252, 146, 114; 251, 106, 74; 239, 59, 44; 203, 24, 29; 165, 15, 21]/255;
colors = [253,174,97;...
  171,217,233;...
  44,123,182;...
  215,25,28;...
  ]/255;
style = {'--', '-', '-.', ':'};
for i = 1:4
  if i == 2
    plot_idx = 1:1:500;
    plot(plot_idx, difference1(i, plot_idx), style{i}, 'color', colors(i,:));
  else
    plot(difference1(i, :), style{i}, 'color', colors(i,:));
  end
end
ylim([-3,1.5])
xlabel('$r$')
ylabel('approximation error')
legend('$\epsilon$-\DP',  '\accu', '\accuc', '\recip',  ...
  'location', 'southeast', 'Interpreter','latex')
legend boxoff
if savefig
  figures_directory = '../neurips2019/figures/';
  if ~isdir(figures_directory), mkdir(figures_directory); end
  figure_name = 'negative_poisson_binomial_approx_r500_error_zoomin';
  matlab2tikz(sprintf('%s/%s.tex', figures_directory, figure_name), ...
    'height',       '\figureheight', ...
    'width',        '\figurewidth', ...
    'parseStrings', false, ...
    'showInfo',     false, ...
    'extraCode',    sprintf('\\tikzsetnextfilename{%s}', figure_name));
end