function [expected_utilities]= two_step_score(...
  problem, train_ind, observed_labels, test_ind, model, weights)

% add another optional output 'top_sum' for bounding usage

num_test   = numel(test_ind);
num_points = size(problem.points, 1);

budget = problem.num_queries ...
  - (numel(train_ind) - problem.num_initial);  % remaining budget 
current_found = search_utility([], [], observed_labels);
if budget <= 0, 
  expected_utilities = current_found * ones(num_test, 1);
  return;
end
% calculate the current posterior probabilities
unlabeled_ind = unlabeled_selector(problem, train_ind, observed_labels);
probabilities = model(problem, train_ind, observed_labels, unlabeled_ind);

success_probabilities = probabilities(:, 1);

reverse_ind = zeros(num_points, 1);
reverse_ind(unlabeled_ind) = 1:numel(unlabeled_ind);

if budget == 1
  expected_utilities = current_found + ...
    success_probabilities(reverse_ind(test_ind));
  return
end

[~, top_ind] = sort(success_probabilities, 'descend');
budget = 1;
weights(train_ind, :) = 0;

expected_utilities = zeros(num_test, 1);
for i = 1:num_test
  this_test_ind = test_ind(i);

  fake_train_ind = [train_ind; this_test_ind];
  
  fake_test_ind = find(weights(:, this_test_ind));
  
  p = success_probabilities;
  p(reverse_ind(this_test_ind)) = 0;
  p(reverse_ind(fake_test_ind)) = 0;
  
  if (isempty(fake_test_ind))
    top_bud_ind = top_ind(1:budget);
    if find(top_bud_ind == reverse_ind(this_test_ind), 1)
      top_bud_ind = top_ind(1:(budget+1));
    end
    baseline = sum(p(top_bud_ind));
    
    expected_utilities(i) = ...
      current_found + ...
      success_probabilities(reverse_ind(this_test_ind)) + ...
      baseline;
    continue;
  end
  
  % sample over labels
  fake_utilities = zeros(problem.num_classes, 1);
  for fake_label = 1:problem.num_classes
    fake_observed_labels = [observed_labels; fake_label];
    
    fake_probabilities = ...
      model(problem, fake_train_ind, fake_observed_labels, ...
      fake_test_ind);
    
    q = sort(fake_probabilities(:, 1), 'descend');
    
    fake_utilities(fake_label) = ...
      current_found     + ...
      (fake_label == 1) + ...
      merge_sort(p, q, top_ind, budget);
  end
  
  % calculate expectation using current probabilities
  expected_utilities(i) = probabilities(...
    reverse_ind(this_test_ind), :) * fake_utilities;
end

end
