function [query_ind, cand_ind, expected_utilities] = ...
  ens_min_cost(...
  problem, train_ind, observed_labels, ~, model, weights, ...
  probability_bound, approx, cutoff, approx_one, adapt)

num_points = size(problem.points, 1);

if ~isfield(problem, 'batch_size')
  problem.batch_size = 1;
end

yet_to_be_found = problem.goal - sum(observed_labels==1);
if cutoff >= 1
  if adapt
    yet_to_be_found = min(yet_to_be_found, cutoff);
  else
    yet_to_be_found = cutoff;
  end
else
  yet_to_be_found = max(round(yet_to_be_found*cutoff), 1);
end

% calculate the current posterior probabilities
unlabeled_ind = unlabeled_selector(problem, train_ind, observed_labels);
probabilities = model(problem, train_ind, observed_labels, unlabeled_ind);

success_probabilities = probabilities(:, 1);

[~, top_ind] = sort(success_probabilities, 'descend');
test_ind = unlabeled_ind(top_ind);
num_test   = numel(test_ind);

if yet_to_be_found < 0
  query_ind = test_ind(1);
  cand_ind = test_ind;
  expected_utilities = zeros(num_test, 1);
  return;
end

reverse_ind = zeros(num_points, 1);
reverse_ind(unlabeled_ind) = 1:numel(unlabeled_ind);

weights(train_ind, :) = 0;

expected_utilities = zeros(num_test, 1);

if strcmp(approx, 'argmin_sum')
  
  cost_func = @merge_sum;
  cost_func_direct = @argmin_sum;
  
elseif contains(approx, '_dp')
    
  cost_func = @(p, q, top_ind, remaining_goal_after_this_point) ...
    compute_negative_poisson_binomial_expectation_dp_approx(...
    p, q, top_ind, remaining_goal_after_this_point, approx_one);
  
  cost_func_direct = @(p, remaining_goal_after_this_point) ...
    compute_negative_poisson_binomial_expectation_dp_approx_direct(...
    p, remaining_goal_after_this_point, approx_one);

end

%% upper bound the score

%% if conditioning on another negative point, current probabilities 
%% are already upper bound
extra = 1e-3;
future_utility_if_neg = -cost_func_direct(success_probabilities(top_ind), ...
  yet_to_be_found) + extra;
%% if conditioning on another positive point
num_positives = 1;
% sorted probability upper bounds
prob_upper_bound = probability_bound(problem, train_ind, ...
  observed_labels, test_ind, num_positives, num_test);
max_num_influence = problem.max_num_influence;
if max_num_influence < yet_to_be_found
  % only those points that could be influenced would change probabilities
  % so we could possibly tighten the bound by using top current
  % probabilities after max_num_influence many points
  cut_rem = num_test - max_num_influence;
  prob_upper_bound = [prob_upper_bound(1:max_num_influence); ...
    min(success_probabilities(top_ind(1:cut_rem)), ...
    prob_upper_bound((max_num_influence+1):end))];
end

future_utility_if_pos = -cost_func_direct(prob_upper_bound, ...
  yet_to_be_found-1) + extra;

upper_bound_of_score = success_probabilities * future_utility_if_pos  + ...
  (1 - success_probabilities) * future_utility_if_neg;
% sort the upper bound in descending order: note the upper bound is a
% monotone function of the probability
upper_bound_of_score = upper_bound_of_score(top_ind);

pruned = false(num_test, 1);
current_max = -problem.num_points;

if isfield(problem, 'do_pruning')
  do_pruning = problem.do_pruning;
else
  do_pruning = true;
end

for i = 1:num_test
  if do_pruning && pruned(i), continue; end

  this_test_ind = test_ind(i);
  
  fake_train_ind = [train_ind; this_test_ind];
  
  fake_test_ind = find(weights(:, this_test_ind));
  
  p = success_probabilities;
  p(reverse_ind(this_test_ind)) = 0;
  p(reverse_ind(fake_test_ind)) = 0;
  
  if (isempty(fake_test_ind))

    this_idx = find(top_ind == reverse_ind(this_test_ind), 1);
    top_ind_wo_this_test = top_ind([1:this_idx-1 this_idx+1:end]);

    util_if_pos = ...
      -cost_func_direct(p(top_ind_wo_this_test), yet_to_be_found-1);
    util_if_neg = ...
      -cost_func_direct(p(top_ind_wo_this_test), yet_to_be_found);
    
    this_prob = success_probabilities(reverse_ind(this_test_ind));
    expected_utilities(i) = ...
      this_prob * util_if_pos + (1-this_prob) * util_if_neg;
  else
    fake_utilities = zeros(problem.num_classes, 1);

    for fake_label = 1:problem.num_classes
      fake_observed_labels = [observed_labels; fake_label];
      fake_probabilities = ...
        model(problem, fake_train_ind, fake_observed_labels, ...
        fake_test_ind);

      remaining_goal_after_this_point = yet_to_be_found-(fake_label==1);

      q = sort(fake_probabilities(:, 1), 'descend');

      fake_utilities(fake_label) = ...
        -cost_func(p, q, top_ind, remaining_goal_after_this_point);
    end
    %% use this implementation to match with batch-ens (numerical issues)
    success_prob = probabilities(reverse_ind(this_test_ind), 1);
    expected_utilities(i) = ...
      [success_prob, 1 - success_prob] * fake_utilities;
    if do_pruning
      if (future_utility_if_pos < fake_utilities(1) || ...
        future_utility_if_neg < fake_utilities(2))
        warning('upper bound bug: upper bound (%f %f), actual (%f, %f)', ...
          future_utility_if_pos, future_utility_if_neg, ...
          fake_utilities(1), fake_utilities(2));
      end
    end
  end
  if expected_utilities(i) > current_max
    current_max = expected_utilities(i);
    query_ind = test_ind(i);
    pruned(upper_bound_of_score < current_max) = true;
  end
end

cand_ind = test_ind(~pruned);
fprintf('\n#train/#total: %d/%d, #pruned/#candidate: %d/%d=%.2f%%\n', ...
  length(train_ind), problem.num_points, ...
  sum(pruned), num_test, mean(pruned)*100);

end
