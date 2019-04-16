% function batch_ind = uncertainty_greedy_combined(problem, train_ind, 
%         observed_labels, test_ind, model, epsilon)
% 
% construct a batch with both exploration and exploitation points. 

% The batch will be composed of points with maximum uncertainties or 
% highest probabilities, the proportion is controlled by a hyperparameter
% epsilon. In particular, batch_size * epsilon portion of the batch will be 
% uncertain points, and the remaining are greedy points (think about 
% epsilon-greedy policy in reinforcement learning). 
%
% Input:
%    epsilon: float numer between 0 and 1.
%         round(batch_size * epsilon) points will be uncertain points.
% Output:
%    batch_ind: indices of the chosen batch
%
% Shali Jiang
% 7/28/2018

function batch_ind = uncertainty_greedy_combined(problem, train_ind, ...
  observed_labels, test_ind, model, epsilon)
% only defined for batch size >= 2
batch_size       = problem.batch_size;
test_probs       = model(problem, train_ind, observed_labels, test_ind);

num_train        = numel(train_ind);
total_budget     = problem.num_queries * batch_size;
budget_spent     = (num_train - problem.num_initial);
remaining_budget = total_budget - budget_spent;

if remaining_budget <= batch_size
  batch_ind = test_ind(1:remaining_budget);  
  return;
end
certainty = abs(test_probs(:,1) - 0.5);  % close to 0 means uncertain
[~, sort_idx1] = sort(certainty);
[~, sort_idx2] = sort(test_probs(:,1), 'descend');

uncertain_amount = max(1, round(batch_size * epsilon));
batch_ind = nan(batch_size, 1);
batch_ind(1:uncertain_amount) = test_ind(sort_idx1(1:uncertain_amount));

greedy_amount = batch_size - uncertain_amount;
k = 1;
j = 1;
while k <= greedy_amount
  greedy_point = test_ind(sort_idx2(j));
  j = j + 1;
  while ismember(greedy_point, batch_ind)
    greedy_point = test_ind(sort_idx2(j));
    j = j + 1;
  end
  batch_ind(uncertain_amount + k) = greedy_point;
  k = k + 1;
end

end
