% ACTIVE_SEARCH_BOUND_SELECTOR selects only potentially optimal search points.
%
% This is an implementation of a selector for the MATLAB Active
% Learning Toolbox:
%
%   https://github.com/rmgarnett/active_learning
%
% that selects only those points that could have optimal l-step
% lookahead expected utility for the active search problem
% (corresponding to search_utility). These points are found by
% exploiting a bound on the future maximum probability after adding a
% given number of positive observations; see the following paper
% for more details:
%
%   R Garnett, Y Krishnamurthy, X Xiong, J Schneider, and R
%   Mann. (2012). Bayesian Optimal Active Search and Surveying. In
%   Proceedings of the 29th International Conference on Machine
%   Learning (ICML 2012). arXiv:1207.6406 [cs.LG]
%
% For more details on the API for probability bounds, see
% expected_search_utility_bound.m.
%
% Usage:
%
%   test_ind = active_search_bound_selector(problem, train_ind, ...
%           observed_labels, model, probability_bound, lookahead)
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
%           lookahead: the number of steps of lookahead to consider
%
% Output:
%
%   test_ind: a list of indices into problem.points indicating the
%             points to consider for labeling
%
% See also EXPECTED_SEARCH_UTILITY_BOUND, SELECTORS, MODELS.

% Copyright (c) 2011--2014 Roman Garnett.

function test_ind = approx_active_search_bound_selector(problem, train_ind, ...
          observed_labels, model, probability_bound, expected_utility, lookahead)

  test_ind = unlabeled_selector(problem, train_ind, []);

  % find point with current maximum posterior probability
  probabilities = model(problem, train_ind, observed_labels, test_ind);
  [p_star, one_step_optimal_ind] = max(probabilities(:, 1));
  one_step_optimal_ind = test_ind(one_step_optimal_ind);
  budget = problem.num_queries ...
    - (numel(train_ind) - problem.num_initial);

  if budget == 1
    test_ind = one_step_optimal_ind;
    return
  end

  % We will need to calculate the expected l-step utility for two
  % points, and we create the required problem structure here.

  % For the selectors, we use this function recursively.
  selectors = cell(lookahead, 1);
  for i = 1:(lookahead - 1)
    selectors{i} = get_selector(@approx_active_search_bound_selector, model, ...
                                probability_bound, expected_utility, i);
  end

  % expected_utility = get_score_function(@approximate_expected_utility, model);

  % find the l-step expected utility of the point with current maximum
  % posterior probability
  p_star_expected_utility = expected_utility_lookahead(problem, ...
          train_ind, observed_labels, one_step_optimal_ind, model, ...
          expected_utility, selectors, lookahead) - ...
      search_utility([], [], observed_labels);

  % find the maximum (l-1)-step expected utility among the
  % currently unlabeled points
  one_fewer_step_optimal_utility = approx_expected_search_utility_bound(problem, ...
          train_ind, observed_labels, test_ind, probability_bound, ...
          lookahead - 1, 0, budget-1);

  % find a bound on the maximum (l-1)-step expected utility after
  % one more positive observation
  
  one_fewer_step_utility_bound = approx_expected_search_utility_bound(problem, ...
          train_ind, observed_labels, test_ind, probability_bound, ...
          lookahead - 1, 1, budget-1);

  % Now a point with probability p can have l-step utility at most
  %
  %        p  * (1 + one_fewer_step_utility_bound  ) +
  %   (1 - p) *      one_fewer_step_optimal_utility
  %
  % and we use this to find a lower bound on p by asserting this
  % quantity must be greater than the l-step expected utility of
  % the point with current maximum probability.
  optimal_lower_bound = ...
    (p_star_expected_utility - one_fewer_step_optimal_utility) / ...
    (1 + one_fewer_step_utility_bound - one_fewer_step_optimal_utility);
  
  cand_ind = probabilities(:, 1) >= min(optimal_lower_bound, p_star);
  
  test_ind = test_ind(cand_ind);
%   if problem.verbose && any(~cand_ind),
%     fprintf('Pruning: %d points are pruned\n', sum(~cand_ind));
%   end


end
