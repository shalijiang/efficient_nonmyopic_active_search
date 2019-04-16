addpath(genpath('./'));

data_dir   = './data';
which_data = {'toy_problem1', 'citeseer_data', 'ecfp1'};
% parameters
data_index          = 1;
data_name           = which_data{data_index};
data_path           = fullfile(data_dir, data_name);

% policies are hard coded with numbers
BATCH_GREEDY   = 1;  % batch greedy policy: choose the point(s) with highest probabilities
BATCH_ENS      = 2;  % batch ENS policy
% sequential simulation with "pessimistic oracle"
SEQ_SIM_ONE_STEP = 13;  % sequential simulation of one-step policy
SEQ_SIM_TWO_STEP = 23; % sequential simulation of two-step policy
SEQ_SIM_ENS = 33;  % sequential simulation of ENS policy

policies            = [BATCH_GREEDY, SEQ_SIM_TWO_STEP, BATCH_ENS];

num_initial         = 1;
num_queries         = 5; % number of batch queries
batch_size          = 1;  % number of points in each batch query
num_experiments     = 5;
num_policies        = length(policies);
verbose             = 0;
total_num_queries   = num_queries * batch_size;
visualize           = 0;

%% load data
[problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
  load_data(data_name);

%% setup problem
problem.num_queries = num_queries;  % note this is the number of batch queries
problem.batch_size  = batch_size;
problem.verbose     = verbose;  % set to true for debugging/verbose output
problem.num_initial = num_initial;
problem.data_name   = data_name;

label_oracle        = get_label_oracle(@lookup_oracle, labels);

%% setup model
model       = get_model(@knn_model, weights, alpha);
model       = get_model(@model_memory_wrapper, model);
num_targets = nan(total_num_queries, num_experiments, num_policies);

for pp = 1:length(policies)
  policy = policies(pp);
  
  if policy == 2 || policy > 30
    tight_level = 4;
    probability_bound = get_probability_bound_improved(...
      @knn_probability_bound_improved, ...
      tight_level, weights, nearest_neighbors', similarities', alpha);
  else
    probability_bound = get_probability_bound(@knn_probability_bound, ...
      weights, full(max(weights)), alpha);
  end
  
  %% setup policy
  [query_strategy, selector] = get_policy(policy, problem, model, ...
    weights, probability_bound);
  
  if visualize
    callback = @(problem, train_ind, observed_labels) ...
      plotting_callback(problem, train_ind, observed_labels, labels);
  else
    callback = @(problem, train_ind, observed_labels) [];
  end
  
  pos_ind = find(labels == 1);
  
  for experiment = 1:num_experiments
    rng(experiment);
    fprintf('\nRunning policy %d experiment %d...\n', policy, experiment);
    
    %% randomly sample num_initial positives as initial training data
    train_ind = [randsample(pos_ind, num_initial)];
    
    observed_labels = labels(train_ind);

    %% run active search cycle for formulated problem
    [chosen_ind, chosen_labels] = active_learning(problem, train_ind, ...
      observed_labels, label_oracle, selector, query_strategy, callback);

    %% collect results
    num_targets(:, experiment, pp) = cumsum(chosen_labels==1);
  end
  
end
%% display average number of targets found
disp(squeeze(mean(num_targets(end, :, :))))
