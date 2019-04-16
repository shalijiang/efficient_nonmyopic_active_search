function [query_ind, cand_ind, expected_utilities] = ...
  ens_with_pruning(...
  problem, train_ind, observed_labels, ~, model, weights, ...
  probability_bound)

num_points = size(problem.points, 1);

if ~isfield(problem, 'batch_size')
  problem.batch_size = 1;
end

remaining_budget = problem.num_queries * problem.batch_size ...
  - (numel(train_ind) - problem.num_initial) ...
  - 1;  % remaining remaining_budget

% calculate the current posterior probabilities
unlabeled_ind = unlabeled_selector(problem, train_ind, observed_labels);
probabilities = model(problem, train_ind, observed_labels, unlabeled_ind);

success_probabilities = probabilities(:, 1);

[~, top_ind] = sort(success_probabilities, 'descend');
test_ind = unlabeled_ind(top_ind);
num_test   = numel(test_ind);

if remaining_budget < 0
  query_ind = test_ind(1);
  cand_ind = test_ind;
  expected_utilities = zeros(num_test, 1);
  return;
end


reverse_ind = zeros(num_points, 1);
reverse_ind(unlabeled_ind) = 1:numel(unlabeled_ind);

weights(train_ind, :) = 0;

expected_utilities = zeros(num_test, 1);

%% upper bound the score
num_positives = 1;

prob_upper_bound = probability_bound(problem, train_ind, ...
  observed_labels, test_ind, num_positives, remaining_budget);

future_utility_if_neg = sum(success_probabilities(...
  top_ind(1:remaining_budget)));

max_num_influence = problem.max_num_influence;
if max_num_influence >= remaining_budget
  future_utility_if_pos = sum(prob_upper_bound(1:remaining_budget));
else
  tmp_ind = top_ind(1:(remaining_budget-max_num_influence));
  future_utility_if_pos = ...
    sum(success_probabilities(tmp_ind)) + ...
    sum(prob_upper_bound(1:max_num_influence));
  %       max_num_influence * prob_upper_bound(1);
end

future_utility = success_probabilities * future_utility_if_pos  + ...
  (1 - success_probabilities) * future_utility_if_neg;

upper_bound_of_score = success_probabilities + future_utility;
upper_bound_of_score = upper_bound_of_score(top_ind);
%%

pruned = false(num_test, 1);
current_max = -1;

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
    top_bud_ind = top_ind(1:remaining_budget);
    if find(top_bud_ind == reverse_ind(this_test_ind), 1)
      top_bud_ind = top_ind(1:(remaining_budget+1));
    end
    baseline = sum(p(top_bud_ind));
    
    expected_utilities(i) = ...
      success_probabilities(reverse_ind(this_test_ind)) + ...
      baseline;
  else
    
    % sample over labels
    fake_utilities = zeros(problem.num_classes, 1);
    for fake_label = 1:problem.num_classes
      fake_observed_labels = [observed_labels; fake_label];
      
      fake_probabilities = ...
        model(problem, fake_train_ind, fake_observed_labels, ...
        fake_test_ind);
      
      q = sort(fake_probabilities(:, 1), 'descend');
      
      fake_utilities(fake_label) = ...
        merge_sort(p, q, top_ind, remaining_budget);
    end
    
    % calculate expectation using current probabilities
    %   expected_utilities(i) = probabilities(...
    %     reverse_ind(this_test_ind), :) * fake_utilities;
    %% use this implementation to match with batch-ens (numerical issues)
    success_prob = probabilities(reverse_ind(this_test_ind), 1);
    expected_utilities(i) = success_prob + ...
      [success_prob, 1 - success_prob] * fake_utilities;
  end
  if expected_utilities(i) > current_max
    current_max = expected_utilities(i);
    query_ind = test_ind(i);
    pruned(upper_bound_of_score < current_max) = true;
  end
end
cand_ind = test_ind(~pruned);
end
