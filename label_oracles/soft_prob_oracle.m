% SOFT_PROB_ORACLE deterministic oracle with given model probabilities.
%
% This provides a label oracle that applies labels that are equivalent to
% the belief of the given model. That is, conditioned on queried point(s),
% the oracle assigns all points a label with a value equal to their 
% calculated probability of being a success.
%
% This oracle is incompatible with models that cannot accomodate noninteger
% label values.
%
% Usage:
%
%   label = argmax_prob_oracle(problem, train_ind, observed_labels,
%                                query_ind, model)
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
%         query_ind: an index into problem.points specifying the
%                    point(s) to be queried
%             model: a function handle to a model to use
%
% Output:
%
%   label: a list of integers between 1 and problem.num_classes
%          with a label that is exactly equal to the model's belief
%          that the label is a success (index 1)
%
% See also LABEL_ORACLES, MULTINOMIAL_ORACLE, MODELS.

function label = soft_prob_oracle(problem, train_ind, observed_labels, ...
          query_ind, model)

  probabilities = model(problem, train_ind, observed_labels, query_ind);

  label = probabilities(:, 1);

end