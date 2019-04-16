% ACTIVE_SEARCH_BOUND_SELECTOR selects only potentially optimal search points.
%
% This is an implementation of a selector for the MATLAB Active
% Learning Toolbox:
%
%   https://github.com/rmgarnett/active_learning
%
% that selects only those points that could have optimal approximate
% expected utility for the active search problem
% (corresponding to search_utility). These points are found by
% exploiting a bound on the future maximum probability after adding a
% given number of positive observations, which is based on the paper:
%
%   R Garnett, Y Krishnamurthy, X Xiong, J Schneider, and R
%   Mann. (2012). Bayesian Optimal Active Search and Surveying. In
%   Proceedings of the 29th International Conference on Machine
%   Learning (ICML 2012). arXiv:1207.6406 [cs.LG]
%
% For more details on the API for probability bounds, see
% knn_probability_bound..
%
% Usage:
%
%   test_ind = approx_exp_bound_selector(problem, train_ind, ...
%           observed_labels, model, probability_bound, num_positives, ...
%		weights, try_m)
%
% Inputs:
%
%             problem: a struct describing the problem, containing fields:
%
%                    points: an (n x d) data matrix for the available points
%               num_classes: the number of classes
%
%           train_ind: a list of indices into problem.points indicating
%                      the thus-far observed points
%     observed_labels: a list of labels corresponding to the
%                      observations in train_ind
%               model: a handle to the probability model to use
%   probability_bound: a handle to the probability bound to use (see
%                      expected_search_utility_bound for details)
%	num_positives: the number of additional positive points 
%			(will be always 1 here)
%	      weights: weight matrix (usually sparse) for knn model
%		try_m: the number of points to compute the score for, 
%			the maximum of which would be the lower bound
%
% Output:
%
%   test_ind: a list of indices into problem.points indicating the
%             points to consider for labeling
%
% See also SELECTORS, MODELS.

% Copyright (c) 2016 Roman Garnett, Shali Jiang.

function test_ind = approx_exp_bound_selector(problem, train_ind, ...
  observed_labels, model, probability_bound, num_positives, weights, try_m)

if ~exist('try_m', 'var'), try_m = 0; end

test_ind = unlabeled_selector(problem, train_ind, []);

% find point with current maximum posterior probability
probabilities = model(problem, train_ind, observed_labels, test_ind);
[p_star, p_star_ind] = max(probabilities(:, 1));
p_star_ind = test_ind(p_star_ind);
if ~isfield(problem, 'batch_size')
  problem.batch_size = 1;
end
budget = problem.num_queries * problem.batch_size ...
  - (numel(train_ind) - problem.num_initial) ...
  - num_positives;  % remaining budget
if budget == 0  % if there is no remaining budget after next point
  test_ind = p_star_ind;  
  return
end

% try m more points to improve the trivial lower bound of the score
try_m_ind = randsample(test_ind, try_m);
lower_bound_ind = [p_star_ind; try_m_ind];

% find the approximate expected utility of a bunch of random points
% and the one with maximum probability
% expectation_bound_if_negative is the sum of top 'budget' probabilities
% since oberving one more negative point can only decrease probabilities
[lower_bound_of_max_score, expectation_bound_if_negative] = ...
  approximate_expected_utility(problem, train_ind, ...
  observed_labels, lower_bound_ind, model, weights);
% the max among these is still a lower bound
[~, max_ind] = max(lower_bound_of_max_score); 
if max_ind ~= 1 
  fprintf('max ind is not max prob ind\n'); 
end
[lower_bound_of_max_score] = max(lower_bound_of_max_score);  
lower_bound_of_max_score = lower_bound_of_max_score - ...
  search_utility([], [], observed_labels);

% Now a point with probability p can have approximate expected utility at most
%
%        p  * (1 + expectation_bound_if_positive) +
%   (1 - p) *      expectation_bound_if_negative 
%
% and we use this to find a lower bound on p by asserting this
% quantity must be greater than the maximum approximate expected utility of
% a random subset of the points including the one with current maximum probability.

% probability upper bound after conditioning on at most 1 more positive point
current_probability_bound = probability_bound(problem, train_ind, ...
  observed_labels, test_ind, num_positives, 1);  % num_positives = 1
expectation_bound_if_positive = current_probability_bound * budget;

optimal_lower_bound = ...  % points with probability lower than this can be pruned
  (lower_bound_of_max_score - expectation_bound_if_negative) / ...
  (1 + expectation_bound_if_positive - expectation_bound_if_negative);

cand_ind = probabilities(:, 1) >= min(optimal_lower_bound, p_star);
test_ind = test_ind(cand_ind);
if problem.verbose
  fprintf('Pruning: %d points are pruned\n', sum(~cand_ind)); 
end

end
