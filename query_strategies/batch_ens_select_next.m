% Hopefully more clever implementation of batch-ens with pruning on the fly
% The following tricks will be used:
%   1. the scores will be computed in descending order of current
%   probabilities
%   2. compute the upper bounds of each point
%   3. constantly updating the global lower bound of the max score, so
%   hopefully more and more points would be pruned as we go
% function estimated_expected_utility = estimate_expected_utility(...
%   problem, ...
%   train_and_selected_ind, observed_labels, test_ind, probs, ...
%   model, weights, ...
%   i, samples, sample_weights, all_probs, ...
%   num_samples, remaining_budget)
%
% Inputs:
%   test_ind: test indices (unlabeled ind) in descending order of probs
%      probs: probabilities of test_ind in descending order
% 9/24/2017

function [point_added_to_batch, cand_ind, estimated_expected_utility, ...
  which_index, upper_bound_of_score] = ...
  batch_ens_select_next(...
  problem, train_and_selected_ind, observed_labels, test_ind, probs, ...
  model, weights, ...
  iter, samples, sample_weights, all_probs, ...
  num_samples, remaining_budget, probability_bound)

num_test   = numel(test_ind);
num_points = size(problem.points, 1);

unlabeled_ind = unlabeled_selector(problem, train_and_selected_ind, ...
  observed_labels);
num_unlabeled = numel(unlabeled_ind);

reverse_ind = zeros(num_points, 1);
reverse_ind(unlabeled_ind) = 1:numel(unlabeled_ind);

weights(train_and_selected_ind, :) = 0;

if isfield(problem, 'do_pruning')
  do_pruning = problem.do_pruning;
else
  do_pruning = true;
end
pruned = false(num_test, 1);
unlabeled_probs = nan(num_unlabeled, num_samples);
top_ind = nan(num_unlabeled, num_samples);
cur_future_utility = zeros(num_samples, 1);
estimated_expected_utility = zeros(num_test, 1);
for j = 1:num_samples
  unlabeled_probs(:,j) = all_probs(unlabeled_ind, j);
  [~, top_ind(:,j)] = sort(unlabeled_probs(:,j), 'descend');
  cur_future_utility(j) = sum(unlabeled_probs(...
    top_ind(1:remaining_budget,j),j));
end

% first compute a global lower bound of the maximum score
future_utility_bound = upper_bound_future_utility(problem, ...
  train_and_selected_ind, observed_labels, samples, sample_weights, iter,...
  num_samples, unlabeled_ind, unlabeled_probs, remaining_budget, ...
  probability_bound, top_ind, cur_future_utility);
upper_bound_of_score = probs + future_utility_bound(reverse_ind(test_ind));

% if compute by descending order of upper bound
if isfield(problem, 'sort_upper') && problem.sort_upper
  [upper_bound_of_score, sort_ind] = sort(upper_bound_of_score, 'descend');
  test_ind = test_ind(sort_ind);
  probs    = probs(sort_ind);
end

% in case sample weights are not normalized
sample_weights = sample_weights ./ sum(sample_weights(1:num_samples));
current_max = -1;
point_added_to_batch = test_ind(1);
num_computed = 0;

memorize = (~isfield(problem, 'memorize') || problem.memorize);
if memorize && 2^(iter-1) > num_samples  % to track repeated samples due to resampling
  memorized = containers.Map();
  new_weights = zeros(size(sample_weights));
  for j = 1:num_samples
    sample_str = sprintf('%d', samples(1:(iter-1), j));
    if isKey(memorized, sample_str)
      ind = memorized(sample_str);
      new_weights(ind) = new_weights(ind) + sample_weights(j);
    else
      memorized(sample_str) = j;
      new_weights(j) = sample_weights(j);
    end
  end
  sample_weights = new_weights / sum(new_weights);
end
for i = 1:num_test
  if do_pruning && pruned(i), continue; end
  %   disp(i);
  this_test_ind = test_ind(i);
  num_computed = num_computed + 1;
  
  if isfield(problem, 'limit') && num_computed > problem.limit
    break
  end
  
  fake_train_ind = [train_and_selected_ind; this_test_ind];
  
  fake_test_ind = find(weights(:, this_test_ind));
  average_future_utility = 0;
  
  
  
  for j = 1:num_samples
    
    if sample_weights(j) < eps, continue; end
    
    p = unlabeled_probs(:, j);
    p(reverse_ind(this_test_ind)) = 0;
    p(reverse_ind(fake_test_ind)) = 0;
    
    if (isempty(fake_test_ind))  % if this point won't affect any other point
      top_bud_ind = top_ind(1:remaining_budget, j);
      if find(top_bud_ind == reverse_ind(this_test_ind), 1)
        % if this point was already in the top sum
        top_bud_ind = top_ind(1:(remaining_budget+1), j);
      end
      future_utility = sum(p(top_bud_ind));
    else
      % sample over labels
      observed_and_sampled = [observed_labels; samples(1:(iter-1), j)];
      fake_utilities = zeros(problem.num_classes, 1);
      for fake_label = 1:problem.num_classes
        fake_observed_labels = [observed_and_sampled; fake_label];
        
        fake_probabilities = ...
          model(problem, fake_train_ind, fake_observed_labels, ...
          fake_test_ind);
        
        q = sort(fake_probabilities(:, 1), 'descend');
        
        fake_utilities(fake_label) = ...
          merge_sort(p, q, top_ind(:,j), remaining_budget);
      end
      
      % calculate expectation using current probabilities
      this_test_prob = unlabeled_probs(reverse_ind(this_test_ind), j);
      future_utility = [this_test_prob, (1-this_test_prob)] * ...
        fake_utilities;
      
    end
    delta_future_utility = future_utility - cur_future_utility(j);
    average_future_utility = average_future_utility + ...
      sample_weights(j)*delta_future_utility;
  end
  estimated_expected_utility(i) = probs(i) + average_future_utility;
  if estimated_expected_utility(i) > current_max
    current_max    = estimated_expected_utility(i);
    
    % index of optimal point in order of probabilities (or upper bounds)
    which_index(1) = i;
    
    % number of points actually computed to get maximum
    % if this is small, we can limit to 
    which_index(2) = num_computed;  
    
    point_added_to_batch   = test_ind(i);
    pruned(upper_bound_of_score <= current_max) = true;
  end
end
cand_ind = test_ind(~pruned);

