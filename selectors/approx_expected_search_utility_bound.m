% EXPECTED_SEARCH_UTILITY_BOUND bounds l-step search expected utility.
%
% This file provides a bound on the l-step lookahead expected utility
% of an unlabeled point for the active search problem (corresponding
% to search_utility). This is accomplished via a function providing a
% bound on the maximum possible posterior probability after adding a
% number of positive observations. This function must satisfy the
% following API:
%
%   bound = probability_bound(problem, train_ind, observed_labels,
%                             test_ind, num_positives)
%
% which should return an upper bound for the maximum posterior
% probability after adding num_positives positive observations.
%
% Usage:
%
%   bound = expected_search_utility_bound(problem, train_ind, observed_labels, ...
%               test_ind, probability_bound, lookahead, num_positives)
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
%            test_ind: a list of indices into problem.points indicating
%                      the points eligible for observation
%   probability_bound: a handle to a probability bound (see above)
%           lookahead: the number of steps of lookahead to consider
%       num_positives: the number of additional positive
%                      observations to consider having added
%
% Output:
%
%   bound: an upper bound for the (lookahead)-step expected count
%          utilities of the unlabeled points
%
% See also ACTIVE_SEARCH_BOUND_SELECTOR.

% Copyright (c) 2011--2014 Roman Garnett.

function bound = approx_expected_search_utility_bound(problem, train_ind, ...
          observed_labels, test_ind, probability_bound, ...
          lookahead, ...
          num_positives, ...
          budget)
  current_probability_bound = probability_bound(problem, train_ind, ...
      observed_labels, test_ind, num_positives, budget);

  if lookahead == 0
    if isscalar(current_probability_bound) 
      bound = budget * current_probability_bound;
    else
      bound = sum(current_probability_bound(1:budget));
    end
  else
    % Otherwise, the bound may be written recursively as follows. Either
    % we get a positive with maximum probability equal to the current
    % bound, or we get a negative. In either case the lookahead
    % decreases by 1.

    upper_bound = @(num_positives) approx_expected_search_utility_bound(problem, ...
            train_ind, observed_labels, test_ind, probability_bound, ...
            lookahead - 1, ...
            num_positives, ...
            budget - 1);

    bound = ...
             current_probability_bound(1)  * (1 + upper_bound(num_positives + 1)) + ...
        (1 - current_probability_bound(1)) *      upper_bound(num_positives);
  end
end
