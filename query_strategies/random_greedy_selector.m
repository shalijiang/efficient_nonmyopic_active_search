function test_ind = random_greedy_selector(problem, train_ind, ~)
% A baseline policy is to spend first half of the budget exploring by
% random selection, and second half exploiting by greedy selection.
% Here we implement this strategy by a "selector", tracking the budget and
% return either a random test point or all unlabeled points.
% 
% 2/3/2017

total_budget = problem.num_queries * problem.batch_size;
budget_used = (numel(train_ind) - problem.num_initial);  

test_ind = unlabeled_selector(problem, train_ind, []);
if budget_used < total_budget/2  % first half of the budget: random
  test_ind = randsample(test_ind, problem.batch_size);
end