function [expected_utilities, top_sum]= approximate_expected_future_utility(...
  problem, train_ind, observed_labels, test_ind, ...
  probabilities, remaining_budget, model, weights)

% add another optional output 'top_sum' for bounding usage

num_test   = numel(test_ind);
num_points = size(problem.points, 1);

% the current posterior probabilities
unlabeled_ind = unlabeled_selector(problem, train_ind, observed_labels);
success_probabilities = probabilities(unlabeled_ind);

[~, top_ind] = sort(success_probabilities, 'descend');

top_sum = sum(success_probabilities(top_ind(1:remaining_budget)));

reverse_ind = zeros(num_points, 1);
reverse_ind(unlabeled_ind) = 1:numel(unlabeled_ind);  

weights(train_ind, :) = 0;

expected_utilities = zeros(num_test, 1);
for i = 1:num_test
  this_test_ind = test_ind(i);

  fake_train_ind = [train_ind; this_test_ind];
  
  fake_test_ind = find(weights(:, this_test_ind));
  
  p = success_probabilities;
  p(reverse_ind(this_test_ind)) = 0;
  p(reverse_ind(fake_test_ind)) = 0;
  
  if (isempty(fake_test_ind))  % if this point won't affect any other point
    top_bud_ind = top_ind(1:remaining_budget);
    if find(top_bud_ind == reverse_ind(this_test_ind), 1)
      % if this point was already in the top sum
      top_bud_ind = top_ind(1:(remaining_budget+1));
    end
    expected_utilities(i) = sum(p(top_bud_ind));
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
      merge_sort(p, q, top_ind, remaining_budget);
  end
  
  % calculate expectation using current probabilities
  expected_utilities(i) = [probabilities(this_test_ind) ...
    (1-probabilities(this_test_ind))] * fake_utilities;
end

end
