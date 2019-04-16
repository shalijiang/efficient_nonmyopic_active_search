function future_utility_bound = upper_bound_future_utility(problem, ...
  train_and_selected_ind, observed_labels, samples, sample_weights, i,...
  num_samples, unlabeled_ind, unlabeled_probs, remaining_budget, ...
  probability_bound, top_ind, cur_future_utility)
%
% assupmtions:
%   unlabeled_probs(:,j) = all_probs(unlabeled_ind, j);
%   [~, top_ind(:,j)] = sort(unlabeled_probs(:,j), 'descend');
%

num_unlabeled = numel(unlabeled_ind);
if size(unlabeled_probs, 1) > num_unlabeled
  top_ind = nan(num_unlabeled, num_samples);
  all_probs = unlabeled_probs;
  unlabeled_probs = nan(num_unlabeled, num_samples);
  for j = 1:num_samples
    unlabeled_probs(:,j) = all_probs(unlabeled_ind, j);
    [~, top_ind(:,j)] = sort(unlabeled_probs(:,j), 'descend');
  end
end
if ~exist('cur_future_utility', 'var')
  cur_future_utility = zeros(num_samples, 1);
end

total_future_utility = zeros(num_unlabeled, 1);

for j = 1:num_samples
  
  observed_and_sampled = [observed_labels; samples(1:(i-1), j)];
  
  num_positives = 1;

  prob_upper_bound = probability_bound(problem, train_and_selected_ind, ...
    observed_and_sampled, unlabeled_ind, num_positives, remaining_budget);
  
  future_utility_if_neg = ...
    sum(unlabeled_probs(top_ind(1:remaining_budget, j), j));
  
  max_num_influence = problem.max_num_influence;
  if max_num_influence >= remaining_budget
    future_utility_if_pos = sum(prob_upper_bound(1:remaining_budget));
  else
    tmp_ind = top_ind(1:(remaining_budget-max_num_influence), j);
    future_utility_if_pos = ...
      sum(unlabeled_probs(tmp_ind, j)) + ...
      sum(prob_upper_bound(1:max_num_influence));
%       max_num_influence * prob_upper_bound(1);
  end

  future_utility = unlabeled_probs(:, j) * future_utility_if_pos  + ...
    (1 - unlabeled_probs(:, j)) * future_utility_if_neg;
  
  delta_future_utility = future_utility - cur_future_utility(j);  % delta
  
  total_future_utility = total_future_utility + ...
    sample_weights(j) * delta_future_utility;
  
end

future_utility_bound = ...
  total_future_utility / sum(sample_weights(1:num_samples));