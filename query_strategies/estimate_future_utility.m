function estimated_future_utility = estimate_future_utility(problem, ...
  train_and_selected_ind, observed_labels, samples, sample_weights, i,...
  num_samples, cand_ind, all_probs, remaining_budget, model, weights)

total_future_utility = zeros(numel(cand_ind), 1);
for j = 1:num_samples
  if problem.verbose
    fprintf(' %dth sample...', j);
  end
  
  observed_and_sampled = [observed_labels; samples(1:(i-1), j)];
  
  future_utility = approximate_expected_future_utility(...
    problem, train_and_selected_ind, observed_and_sampled, cand_ind, ...
    all_probs(:,j), remaining_budget, model, weights);
  
  total_future_utility = total_future_utility + ...
    sample_weights(j)*future_utility;
  
end
if problem.verbose, fprintf('\n'); end

estimated_future_utility = ...
  total_future_utility / sum(sample_weights(1:num_samples));