addpath(genpath('./'));
% also add "active_learning" and "active_search" to path
% addpath(genpath('path_to_active_learning')); %https://github.com/rmgarnett/active_learning.git
% addpath(genpath('path_to_active_search')); %https://github.com/rmgarnett/active_search.git

data_dir   = './data';
which_data = {'toy_problem0', 'toy_problem1', 'citeseer_data', 'ecfp1'};
% To run for citeseer_data: run ./data/citeseer/prepare_venue_subgraph.m to
% produce the data first
% To run for drug discovery data: follow intructions from https://github.com/rmgarnett/active_virtual_screening.git
% generate the target_xxx_ecfp4_nearest_neighbors_xxx.mat and put them
% under ./data/ecfp4/
data_index          = 2;  
data_name           = which_data{data_index};

policy_codes  % defines policies coded by constant numbers
policies            = [GREEDY, SS_TWO_0, BATCH_ENS];

% set this to 1 if you want to plot the selected point (2D problem only)
visualize           = 0;
% set this to 1 if you want to print info for every iteration
verbose             = 1;

% if batch_size = 1, perform fully sequential active search
batch_size          = 1;  % number of points in each batch query
num_queries         = 20; % number of batch queries
total_num_queries   = num_queries * batch_size;

num_initial         = 1;  % number of initial positive training points 

num_experiments     = 3;  % number of experiments to repeat 
num_policies        = length(policies);

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
  
  %% set up the function of bounding the probabilities
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
    train_ind = randsample(pos_ind, num_initial);
    
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
