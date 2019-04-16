function id_str = get_id_str(data_name, policies, num_queries, ...
  batch_size, experiments, max_num_samples, resample, limit, num_initial)
if (~exist('num_initial', 'var') || isempty(num_initial))
  num_initial = 1;  % default: don't sort by upper_bound
end
p = 'policy_';
for i = policies
  p = [p num2str(i) '_'];
end

if ismember(2, policies)
  p = [p 'no_seed_'];
  if ~exist('limit', 'var') || limit == Inf
    id_str = sprintf('%s_%sns_%d_re_%d_nq_%d_bs_%d_exp%d_%d', ...
      data_name, p, max_num_samples, resample, num_queries, batch_size, ...
      experiments(1), experiments(end));
  else
    id_str = sprintf('%s_%sns_%d_re_%d_lim_%d_nq_%d_bs_%d_exp%d_%d', ...
      data_name, p, max_num_samples, resample, limit, num_queries, batch_size, ...
      experiments(1), experiments(end));
  end
  
else
  id_str = sprintf('%s_%snq_%d_bs_%d_exp%d_%d', ...
    data_name, p, num_queries, batch_size, ...
    experiments(1), experiments(end));
end

if num_initial > 1
  id_str = sprintf('%s_ni_%d', id_str, num_initial);
end


