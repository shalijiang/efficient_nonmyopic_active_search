% function query_ind = batch_ens(problem, train_ind, observed_labels, ...
%  test_ind, batch_size, model, weights, selector)
%
% This is an approximate implementation of the batch-ens policy
% for batch active search:
%
%   f(S) = \sum_{x\in S} Pr(y=1|x) + E_{y_S}[\sum'_{B-b-|D|}
%                 Pr(y'=1|x',D\cup y_S)]
%      S = argmax_{S \in X, |S| <= b} f(S)          (***)
%
% Basically batch-ens is an extension of the sequential ens policy as
% presented in
%   Jiang et al., "Efficient nonmyopic active search", ICML 2017.
% The utility of a set S (size no larger than batch size)
% is composed of two parts:
%   (1) the expected utility of points in S (sum of probs)
%   (2) the future expected utility after this whole batch conditioned on
%   current observations and points currently in the batch (take
%   expectation over the joint distribution of labels of points in S)
%
% Policy (***) is intractable for two reasons:
%   (1) there are n choose b possible subsets of size b to consider
%   (2) for each subset S, the label space of S is of size 2^b
% So exact implementation of (***) would require O(n^b 2^b) complexity.
%
% We approximate batch-ens in the following way:
%   (1) we conjecture f(S) is a submodular set function (needs proving), so
%   we greedily construct S:
%     for i = 1..b, x_i = argmax_{x} f(S_{i-1} U {x}) - f(S_{i-1})
%   If f(S) is really submodular, this greedy solution would have nice
%   guarantee (Nemhauser et al. 1978).
%   (2) for large batch size, we approximate the expected future utility
%   part by Monte Carlo simulation:
%     We use a small number of samples from the posterior of y_S given D
%     (use Gibbs-like sampling), then use the empricial average as the
%     expectation.
%
% Input:
%           problem: problem structure with fields as follows:
%       .num_points: total number of points
%      .num_initial: number of initial observations
%      .num_queries: budget
%       .batch_size: number of points in each batch query
%
%         train_ind: indices of current observations [col vec]
%   observed_labels: observed labels corresponding to train_ind [col vec]
%          test_ind: indices of unlabeled point to choose the batch from
%        batch_size: number of points to choose for the batch [integer]
%             model: probabilistic model giving posterior probs
%           weights: weights of knn model
%          selector: pruning function for each iteration of batch construction
%       num_samples: number of samples for approximating the expected future
%   utility
%
% Output:
%         batch_ind: indices of the chosen batch
%
% Shali Jiang
% 9/14/2017

function batch_ind = batch_ens(problem, train_ind, observed_labels, ...
  test_ind, model, weights, probability_bound, max_num_samples, lookahead)

num_points       = problem.num_points;
num_classes      = problem.num_classes;
batch_size       = problem.batch_size;
verbose          = problem.verbose;
probs            = zeros(num_points, 1);
probs(train_ind) = observed_labels == 1;  % assume target class label=1
test_probs       = model(problem, train_ind, observed_labels, test_ind);

[test_probs, sort_ind] = sort(test_probs(:,1), 'descend');
test_ind         = test_ind(sort_ind);
probs(test_ind)  = test_probs;
num_train        = numel(train_ind);
remaining_budget = problem.num_queries * batch_size...
  - (num_train - problem.num_initial);

if remaining_budget <= batch_size
  batch_ind = test_ind(1:remaining_budget);  % test_ind is sorted by probs.
  return;
end

remaining_budget_after_this_batch = remaining_budget  - batch_size;

if ~exist('lookahead', 'var'), lookahead = Inf; end

if lookahead < 0
  error('Argument ''lookahead'' of function %s should be nonnegative', ...
    mfilename);
elseif lookahead == 0  % no lookahead: greedy batch
  % require: test_ind is sorted by probs.
  batch_ind = test_ind(1:min(remaining_budget, batch_size));  
  return;
elseif lookahead >= num_points - num_train - batch_size  
  % lookahead is the remaining budget: ENS
  % to do ENS, easiest way is to set this to Inf
  next_batch_size = remaining_budget_after_this_batch;
elseif lookahead < 1  % lookahead is some portion of the remaining budget
  % note this is still in cognizant of the budget
  % e.g. if lookahead=0.5, only lookahead half of the remaining budget
  next_batch_size = round(remaining_budget_after_this_batch * lookahead);
else  % constant lookahead 
  % e.g. if lookahead = 1, then it's batch-two-step policy
  next_batch_size = round(batch_size * lookahead);
  next_batch_size = min(remaining_budget_after_this_batch, next_batch_size);
end
  
if next_batch_size == 0  % if no next batch, do greedy
  % require: test_ind is sorted by probs.
  batch_ind = test_ind(1:min(remaining_budget, batch_size));
  return;
end

batch_ind        = zeros(batch_size, 1);
samples          = nan(batch_size, max_num_samples);  % each column is a sample
sample_weights   = ones(1, max_num_samples);
all_probs        = repmat(probs, 1, max_num_samples);

weights(train_ind, :) = 0;
test_pruning = isfield(problem, 'test_pruning') && problem.test_pruning;
test_memo = isfield(problem, 'test_memo') && problem.test_memo;
save_score = isfield(problem, 'save_score') && problem.save_score;

if save_score
  all_estimates = zeros(num_points, batch_size, 3);
end
for i = 1:batch_size
  
  train_and_selected_ind = [train_ind; batch_ind(1:(i-1))];
  num_samples = min(2^(i-1), max_num_samples);

  
  tt = tic;
  [chosen_ind, cand_ind, estimates, which_ind, upper_bound_of_score] = ...
    batch_ens_select_next(...
    problem, train_and_selected_ind, observed_labels, test_ind, test_probs, ...
    model, weights, ...
    i, samples, sample_weights, all_probs, ...
    num_samples, next_batch_size, probability_bound);
  time = toc(tt);
  
  if test_pruning
    problem.do_pruning = 0;
    tt0 = tic;
    [chosen_ind0, ~, estimates, which_ind, upper_bound_of_score] = ...
      batch_ens_select_next(...
      problem, train_and_selected_ind, observed_labels, test_ind, test_probs, ...
      model, weights, ...
      i, samples, sample_weights, all_probs, ...
      num_samples, next_batch_size, probability_bound);
    time0 = toc(tt0);
    problem.do_pruning = 1;
    chose_same = (chosen_ind == chosen_ind0);
    fprintf('pruning: %.2f vs. %.2f, chose same: %d\n', ...
      time, time0, chose_same)
  end
  
  if test_memo && 2^(i-1) > max_num_samples
    problem.memorize = 0;
    tt0 = tic;
    [chosen_ind0, ~, estimates1, which_ind, upper_bound_of_score] = ...
      batch_ens_select_next(...
      problem, train_and_selected_ind, observed_labels, test_ind, test_probs, ...
      model, weights, ...
      i, samples, sample_weights, all_probs, ...
      num_samples, next_batch_size, probability_bound);
    time0 = toc(tt0);
    problem.memorize = 1;
    chose_same = (chosen_ind == chosen_ind0);
    fprintf('memorize: %.2f vs. %.2f (memo), chose same: %d estimates diff: %f\n', ...
      time, time0, chose_same, norm(estimates-estimates1))
  end
  
  if save_score
    all_estimates(test_ind, i, 1) = estimates;
    all_estimates(test_ind, i, 2) = upper_bound_of_score;
    all_estimates(test_ind, i, 3) = test_probs;
  end
  
  if verbose
    fprintf('remaining budget after this batch %d, %d / %d selected from %d points (%d after pruning) in %.2f sec.\n', ...
      remaining_budget_after_this_batch, i, batch_size, numel(test_ind), numel(cand_ind), time);
  end
  
  chosen_test_ind_ind = find(test_ind == chosen_ind, 1);
  test_ind(chosen_test_ind_ind) = [];
  test_probs(chosen_test_ind_ind) = [];
  
  batch_ind(i) = chosen_ind;
  
  %% add this when all experiments are complete, 
  %% otherwise the we can't reproduce the results since 
  %% adding this would change the psuedorandom sequence.
%   if i == batch_size 
%     continue
%   end
  
  % find the points that can be affected by the chosen point
  weights(chosen_ind, :) = 0;
  updating_ind = find(weights(:, chosen_ind));
  
  %% sample for this chosen point conditioned on existing samples
  if num_samples * 2 <= max_num_samples  % double the samples
    sample_weights0 = sample_weights;
    for j = num_samples:-1:1
      % probability of positive/negative
      both_probs = [all_probs(chosen_ind, j); ...
        1- all_probs(chosen_ind, j)];
      
      for fake_label = 1:num_classes  % assume num_classes = 2
        sample_idx = 2*j - 2 + fake_label;
        samples(1:i, sample_idx) = ...
          [samples(1:(i-1), j); fake_label];
        
        % use chain rule to update the weights of the samples
        sample_weights(sample_idx) = sample_weights0(j) * both_probs(fake_label);
        
        observed_and_sampled = [observed_labels; ...  % points already observed
          samples(1:(i-1), j); ...  % points already in the batch
          fake_label];  % newly added point
        
        updated_probs = model(problem, ...
          [train_and_selected_ind; chosen_ind], ...
          observed_and_sampled, updating_ind);
        all_probs(:, sample_idx) = all_probs(:,j);
        all_probs(updating_ind, sample_idx) = updated_probs(:,1);
      end
    end
    num_samples = num_samples * 2;
    sample_weights = sample_weights / sum(sample_weights(1:num_samples));
  else  % 2^i exceeds the max number of samples specified
    % add one sample of the chosen point conditioned on each existing samples
    
    % resample with replacement in case one sample has overwhelming weight
    resample = (~isfield(problem, 'resample') || problem.resample);
    %rng(length(train_ind)+i);  % will this introduce bias?
    if resample && 2^(i-1) <= max_num_samples 
      
      resample_ind = randsample(1:num_samples, max_num_samples, true, ...
        sample_weights);
      sample_weights = ones(1, max_num_samples) / max_num_samples;
      samples   = samples(:, resample_ind);
      all_probs = all_probs(:, resample_ind);
    end
    for j = 1:num_samples
      fake_label = randsample(num_classes, 1, true, ...
        [all_probs(chosen_ind, j); 1 - all_probs(chosen_ind, j)]);
      
      observed_and_sampled = [observed_labels; ...  % points already observed
        samples(1:(i-1), j); ...  % points already in the batch
        fake_label];  % newly added point
      updated_probs = model(problem, ...
        [train_and_selected_ind; chosen_ind], ...
        observed_and_sampled, updating_ind);
      
      all_probs(updating_ind, j) = updated_probs(:,1);
      
      samples(i, j) = fake_label;
    end
  end
end
if save_score
  save(sprintf('%s/%s_%d_%d_scores', problem.result_dir, problem.data_name, ...
    remaining_budget_after_this_batch, batch_size), 'all_estimates');
end
end
