function [query_strategy, selector] = get_policy(policy, problem, ...
  model, weights, probability_bound, max_num_samples)

if isnumeric(policy)
  policy = get_policy_struct(policy);
end

selector    = get_selector(@unlabeled_selector);
batch_size  = problem.batch_size;
policy_name = policy.name;

if ismember(policy_name, {'greedy', 'random-greedy'})
  
  score_function = @(problem, train_ind, observed_labels, test_ind) ...
    search_expected_utility(problem, train_ind, observed_labels, ...
    test_ind, model);
  query_strategy = get_query_strategy(@argmax, score_function, ...
    batch_size);
  if strcmp(policy_name, 'random-greedy')
    selector    = get_selector(@random_greedy_selector);
  end
  
elseif  contains(policy_name, 'batch-')
  % max_num_samples is only for batch_ens
  if ~exist('max_num_samples', 'var') || isempty(max_num_samples)
    max_num_samples = 16;
  end
  if contains(policy_name, 'batch-ens')
    if strcmp(policy_name, 'batch-ens')
      lookahead = Inf;
    else
      lookahead = str2double(policy_name(11:end));
    end
  elseif contains(policy_name, 'batch-two-step')
    if strcmp(policy_name, 'batch-two-step')
      lookahead = 1;
    else
      lookahead = str2double(policy_name(16:end));
    end
  else
    error('In %s: unknown policy name', mfilename);
  end
  query_strategy = get_query_strategy(@batch_ens, ...
    model, weights, probability_bound, max_num_samples, lookahead);
  
elseif strcmp(policy_name, 'sequential')
  % policy should have following fields:
  %  .name
  %  .subname
  %  .fiction_label
  %  .selector

  subpolicy_name = policy.subname;
  
  switch subpolicy_name
    
    case 'myopic'
      lookahead = policy.lookahead;
      selectors = cell(lookahead, 1);
      for i = 1:lookahead
        selectors{i} = get_selector(@active_search_bound_selector, ...
          model, probability_bound, i);
      end
      fiction_selector = selectors{lookahead};
      if lookahead == 1
        expected_utility = get_score_function(@search_expected_utility, ...
          model);
        score_function = get_score_function(@expected_utility_lookahead, ...
          model, expected_utility, selectors, lookahead);
      else
        score_function = get_score_function(@two_step_score, model, weights);
      end
      sub_query_strategy = get_query_strategy(@argmax, score_function);
      
    case 'ens'
      fiction_selector = get_selector(@unlabeled_selector);
      sub_query_strategy = get_query_strategy(@ens_with_pruning, ...
        model, weights, probability_bound);
      
    case 'ims'
      score_function = get_score_function(@kdd13, model, weights);
      sub_query_strategy = get_query_strategy(@argmax, score_function);
      fiction_selector = @unlabeled_selector;
  end
  
  fiction_oracle = policy.fiction_oracle;
  switch fiction_oracle
    case 'sample'
      fiction_oracle = get_label_oracle(@probabilistic_oracle, model);
    case 'argmax'
      fiction_oracle = get_label_oracle(@argmax_prob_oracle, model);
    case 'always0'
      fiction_oracle = get_label_oracle(@fixed_oracle, 0);
    case 'always1'
      fiction_oracle = get_label_oracle(@fixed_oracle, 1);
  end
  
  query_strategy = get_query_strategy(@sequential_simulation_batch_iter,...
    batch_size, sub_query_strategy, fiction_oracle, fiction_selector);

elseif strcmp(policy_name, 'uncertainty-greedy')
  query_strategy = get_query_strategy(@uncertainty_greedy, ...
        model, policy.transition_ratio);
elseif strcmp(policy_name, 'uncertainty-greedy-mixed')
  query_strategy = get_query_strategy(@uncertainty_greedy_combined, ...
        model, policy.epsilon);
else
  error('In %s: unknown policy name', mfilename);
end
