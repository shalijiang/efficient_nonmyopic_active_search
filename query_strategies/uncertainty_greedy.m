% function batch_ind = uncertainty_greedy(problem, train_ind, 
%         observed_labels, test_ind, model, transition_ratio)
% 
% heuristic exploration/exploitation transition, first do active learning 
% by uncertainty sampling, then do greedy sampling; transition is
% controlled by a hyperparameter transition_ratio; 
% e.g. when transition_ratio = 0.5, then perform active learning in the 
% first half of the budget, and greedy search in the second
% 
% Input:
%    transition_ratio: transition_ratio * total_budget would be the
%                      transition point
% Output:
%    batch_ind: indices of the chosen batch
%
% Shali Jiang
% 7/28/2018

function batch_ind = uncertainty_greedy(problem, train_ind, ...
  observed_labels, test_ind, model, transition_ratio)

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

if budget_spent <= transition_ratio * total_budget
  % uncertainty sampling 
  certainty = abs(test_probs(:,1) - 0.5);  % close to 0 means uncertain
  [~, sort_idx] = sort(certainty);
else 
  % if remaining budget is small, greedy sampling 
  [~, sort_idx] = sort(test_probs(:,1), 'descend');
end

batch_ind = test_ind(sort_idx(1:batch_size));  
end
