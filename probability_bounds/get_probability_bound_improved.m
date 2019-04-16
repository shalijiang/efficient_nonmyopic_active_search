% GET_PROBABILITY_BOUND creates a function handle to a probabilitybound.
%
% This is a convenience function for easily creating a function handle
% to a probability boundr. Given a handle to a probability bound and
% its additional arguments (if any), returns a function handle for use
% in, e.g., expected_search_utility_bound.m.
%
% Example:
%
%   probability_bound = get_probability_bound(@knn_probability_bound, ...
%           weights, max(weights), alpha);
%
% returns the following function handle:
%
%   @(problem, train_ind, observed_labels, test_ind, num_positives) ...
%       knn_probability_bound(problem, train_ind, observed_labels, ...
%                             test_ind, num_positives, weights, ...
%                             max(weights), alpha)
%
% This is primarily for improving code readability by avoiding
% repeated verbose function handle declarations.
%
% Usage:
%
%   probability_bound = get_probability_bound(probability_bound, varargin)
%
% Inputs:
%
%   probability_bound: a handle to the desired probability bound
%            varargin: any additional inputs to be bound to the
%                      probability_bound beyond those required by the
%                      standard interface (problem, train_ind,
%                      observed_labels, test_ind, num_positives)
%
% Output:
%
%   probability_bound: a function handle to the desired probability
%                      bound for use in expected_search_utility_bound

% Copyright (c) 2014 Roman Garnett.

function probability_bound = get_probability_bound_improved(probability_bound, varargin)

  probability_bound = ...
      @(problem, train_ind, observed_labels, test_ind, num_positives, budget) ...
      probability_bound(problem, train_ind, observed_labels, test_ind, ...
                        num_positives, budget, varargin{:});

end