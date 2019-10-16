% UNLABELED_SELECTOR selects points not yet observed.
%
% Usage:
%
%   test_ind = unlabeled_selector(problem, train_ind, observed_labels)
%
% Inputs:
%
%           problem: a struct describing the problem, which must at
%                    least contain the field:
%
%              points: an (n x d) data matrix for the avilable points
%
%         train_ind: a list of indices into problem.points indicating
%                    the thus-far observed points
%   observed_labels: a list of labels corresponding to the
%                    observations in train_ind
%
%                    Note: this input, part of the standard selector
%                    API, is ignored by unlabeled_selector. If
%                    desired, for standalone use it can be replaced by
%                    an empty matrix.
%
% Output:
%
%   test_ind: a list of indices into problem.points indicating the
%             points to consider for labeling
%
% See also SELECTORS.

% Copyright (c) 2013--2014 Roman Garnett.

function test_ind = unlabeled_random_selector1(problem, train_ind, select_size)

  test_ind = identity_selector(problem, [], []);
  test_ind(train_ind) = [];
  if nargin < 3, return; end
  n = length(test_ind);
  if select_size > n
      select_size = n;
      warning('select_size %d larger than number of test points %s', ...
          select_size, n);
  end
  if select_size < length(test_ind)
    test_ind = randsample(test_ind, select_size);
  end

end