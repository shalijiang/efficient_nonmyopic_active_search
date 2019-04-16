% KNN_PROBABILITY_BOUND probability bound for a k-NN classifier.
%
% This function provides a bound for
%
%   \max_i p(y_i = 1 | x_i, D)
%
% after adding additional points to the current training set for a
% k-NN classifier (as implemented in knn_model.m). This bound is
% intended to be used with expected_seach_utility_bound.
%
% Usage:
%
%   bound = knn_probability_bound(responses, train_ind, test_ind, ...
%           weights, max_weights, pseudocount, num_positives)
%
% Inputs:
%
%           problem: a struct describing the problem, containing the
%                    fields:
%
%                  points: an (n x d) data matrix for the avilable points
%             num_classes: the number of classes
%
%         train_ind: a list of indices into problem.points indicating
%                    the thus-far observed points
%   observed_labels: a list of labels corresponding to the
%                    observations in train_ind
%          test_ind: a list of indices into problem.points indicating
%                    the test points
%           weights: an (n x n) matrix of weights
%       max_weights: precomputed max(weights)
%             alpha: the hyperparameter vector \alpha
%                    (1 x problem.num_classes)
%
% Output:
%
%   bound: an upper bound for the maximum posterior probability after
%          adding num_positives positive observations for the points
%          in test_ind.
%
% See also KNN_MODEL, EXPECTED_SEARCH_UTILITY_BOUND, ACTIVE_SEARCH_BOUND_SELECTOR.

% Copyright (c) 2011--2014 Roman Garnett.

function bound = knn_probability_bound_improved(~, train_ind, observed_labels, ...
  test_ind, num_positives, budget, tight_level, weights, knn_ind, knn_weights, alpha)
% add two more parameters than the 'knn_probability_bound':
%    budget, tight_level
% budget: will return the top 'budget' probability bound
% tight_level (1,2 or 3):
%    1: use the max_weight for all test point
%    2: use each test_point's own max_weight
%    3: use each test_point's own top 'num_positives' weights
% Note tight_level 3 should provide the tightest bound, but also more
% expansive to compute
  if nargin < 6 || ~exist('budget', 'var'), budget = 1; end
  if ~exist('tight_level', 'var'), tight_level = 3; end
  % transform observed_labels to handle multi-class
  positive_ind = (observed_labels == 1);
  successes = sum(weights(test_ind, train_ind( positive_ind)), 2);
  failures  = sum(weights(test_ind, train_ind(~positive_ind)), 2);

  if num_positives == 0
    success_count_bound = 0;
  elseif tight_level == 1  % use one max max weight for all test_ind
    success_count_bound = max(knn_weights(test_ind, 1)) * num_positives;  % scalar

  elseif tight_level == 2  % multiply each test ind's own max

    max_weights = knn_weights(test_ind, 1);
    success_count_bound = max_weights * num_positives;  % column vector

  elseif tight_level == 3  % use each test ind's own top num_positives

    top_weights = knn_weights(test_ind, 1:num_positives);
    success_count_bound = sum(top_weights, 2);  % column vector

  elseif tight_level == 4  % use top num_positives, excluding train_ind

    in_train = ismember(knn_ind(:), train_ind);
    knn_weights(in_train) = 0;
    if num_positives == 1
      success_count_bound = max(knn_weights(test_ind, ...
        1:min(end, length(train_ind)+num_positives)), [], 2);  % column vector

    else
      findrow = @(row) find(row, num_positives, 'first');
      n = size(knn_weights, 1);
      top_ind = cell2mat(cellfun(findrow, num2cell(knn_weights, 2), ...
        'uniformoutput',false))';  % num_positives by n matrix 
      row_index = kron((1:n)', ones(num_positives, 1));
      linear_ind = sub2ind(size(knn_weights), row_index, top_ind(:));
      top_weights = reshape(knn_weights(linear_ind), [num_positives, n]);
%       knn_weights = sort(knn_weights(test_ind, ...
%         1:min(end, length(train_ind)+num_positives)), 2, 'descend');
%       top_weights = knn_weights(:, 1:num_positives);
      success_count_bound = sum(top_weights(:, test_ind), 1)';  % column vector
    end
  end

  max_alpha = (    alpha(1)      + successes + success_count_bound);
  min_beta  = (sum(alpha(2:end)) + failures);

  probabilities = max_alpha ./ (max_alpha + min_beta);
  if budget <= 1
    bound = max(probabilities);
  else
    probabilities = sort(probabilities, 'descend');
    bound = probabilities(1:budget);
  end

end