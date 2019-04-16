function [scores]= kdd13(...
  problem, train_ind, observed_labels, test_ind, model, weights, alpha)
%
% This is the score function for active search proposed in:
%
% Xuezhi Wang, Roman Garnett, Jeff Schneider, 
% "Active search on graphs", KDD 2013.
%
% The score function is
%
%   Pr(y=1|x, D) + alpha * IM(x)
%
% where "IM" is the impact factor measuring the impact of the candidate
% point by the total change of the probability. That is, 
%   IM(x) = Pr(y=1|x,D)*sum_i(Pr(y_i=1|x_i,D,(x,y=1)) - Pr(y_i=1|x_i,D))
% Note observing a new positive point usually only change the probabilities
% of a small fraction of all the unlabeled points.
% 
% Assume "test_ind" is same as "unlabeled_ind"
%
% Shali Jiang, 1/26/2017

if nargin < 7
  alpha = 0.0001;  % reasonable choice according to the paper
end
num_points = size(problem.points, 1);

% calculate the current posterior probabilities
unlabeled_ind = unlabeled_selector(problem, train_ind, observed_labels);
assert(isequal(test_ind, unlabeled_ind), 'test_ind not same as unlabeled');
probabilities = model(problem, train_ind, observed_labels, unlabeled_ind);
 
current_probabilities = probabilities(:, 1);

reverse_ind = zeros(num_points, 1);
reverse_ind(unlabeled_ind) = 1:numel(unlabeled_ind);

weights(train_ind, :) = 0;

num_unlabeled = numel(unlabeled_ind);
scores = zeros(num_unlabeled, 1);
for i = 1:num_unlabeled
  this_test_ind = unlabeled_ind(i);
  fake_train_ind = [train_ind; this_test_ind];
  % only consider the postive case
  fake_observed_labels = [observed_labels; 1];  
  % only compute probabilities for those that will change
  fake_test_ind = find(weights(:, this_test_ind));  
  prob_of_this_test_ind = current_probabilities(reverse_ind(this_test_ind));

  fake_probabilities = ...
      model(problem, fake_train_ind, fake_observed_labels, fake_test_ind);
  new_probabilities = fake_probabilities(:,1);
  old_probabilities = current_probabilities(reverse_ind(fake_test_ind));
  sum_prob_diff = sum(new_probabilities - old_probabilities);
  IM = prob_of_this_test_ind * sum_prob_diff;
  scores(i) = prob_of_this_test_ind + alpha * IM;
end

end
