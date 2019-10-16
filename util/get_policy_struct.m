function policy = get_policy_struct(policy_num)
% create a policy struct based on an integer policy_num

if policy_num == 1
  policy.name = 'greedy';
elseif policy_num == 2
  policy.name = 'batch-ens';
elseif policy_num > 2 && policy_num < 3
  portion = policy_num - 2;
  % this parameter has different interpretation in different settings:
  % in budgeted setting, portion should be in (0,1), and it is iterpreted
  % as setting the remaining budget as this portion of actual remaining budget
  % in cost effective setting, portion should be multiplied by 10, then 
  % the remaining budget is estimated by multiplying the remaining goal 
  policy.name = sprintf('batch-ens-%g', portion);
elseif policy_num == 3
  policy.name = 'random-greedy';
elseif policy_num == 4
  policy.name = 'batch-two-step';
elseif policy_num > 4 && policy_num < 5  % 4.xy means next_batch_size = yx 
  % trailing zeros is meaningless
  policy_str = num2str(policy_num);
  next_batch_size = str2double(policy_str(end:-1:3));
  policy.name = sprintf('batch-two-step-%d', next_batch_size);
elseif policy_num >= 5 && policy_num < 6 % cost 
  policy.name = 'cens';
  policy.approx = 'exact_dp';
  policy.cutoff = Inf;
  policy.adapt = true;  % whether to adapt to remaining target after cutoff
  policy.num_samples = 1000;
  policy.approx_one = 1-1e-9;
  if policy_num >= 5.1 && policy_num < 5.2
    policy.approx = 'argmin_sum';
    if policy_num > 5.1  % fixed cutoff
      policy_str = num2str(policy_num);
      policy.cutoff = str2double(policy_str(end:-1:4));
    end
  elseif policy_num >= 5.2 && policy_num < 5.3
    policy.approx = 'argmin_sum';  % cutoff proportional to remaining goal
    portion = (policy_num - 5.2)*10;  
    if portion > 0
      policy.cutoff = portion;
    end
  
  elseif policy_num >= 5.3 && policy_num < 5.4
    policy.approx = 'approx_dp';
    policy.approx_one = 1-1e-6;
    if policy_num > 5.3  % fixed cutoff
      policy_str = num2str(policy_num);
      policy.cutoff = str2double(policy_str(end:-1:4));
    end
  elseif policy_num >= 5.4 && policy_num < 5.5
    policy.approx = 'approx_dp';  % cutoff proportional to remaining goal
    policy.approx_one = 1-1e-6;
    portion = (policy_num - 5.4)*10;  
    if portion > 0
      policy.cutoff = portion;
    end
  elseif policy_num >= 5.5 && policy_num < 5.6
    policy.approx = 'exact_dp';
    policy.approx_one = 1;
    if policy_num > 5.3  % fixed cutoff
      policy_str = num2str(policy_num);
      policy.cutoff = str2double(policy_str(end:-1:4));
    end
  elseif policy_num >= 5.6 && policy_num < 5.7
    policy.approx = 'exact_dp';  % cutoff proportional to remaining goal
    policy.approx_one = 1;
    portion = (policy_num - 5.2)*10;  
    if portion > 0
      policy.cutoff = portion;
    end
  elseif policy_num >= 5.7 && policy_num < 5.8
    policy.approx = 'monte_carlo'; 
    policy_str = num2str(policy_num);
    policy.cutoff = str2double(policy_str(end:-1:4));
    policy.num_samples = 1000;
  elseif policy_num >= 5.8 && policy_num < 5.9
    % to test whether adaptation to remaining targets helps
    policy.approx = 'argmin_sum';
    if policy_num > 5.8  % fixed cutoff
      policy_str = num2str(policy_num);
      policy.cutoff = str2double(policy_str(end:-1:4));
    end
    policy.adapt = false;
  end
elseif policy_num > 6 && policy_num < 7
  policy.transition_ratio = policy_num - 6;
  policy.name = 'uncertainty-greedy';
elseif policy_num >= 7 && policy_num < 8
  policy.epsilon = policy_num - 7;
  policy.name = 'uncertainty-greedy-mixed';
else
  % two digits interger "xy" in {1:3} \times {1:5}
  % x: for sequential policy
  %  1: one-step
  %  2: two-step
  %  3: ens
  %  4: kdd13
  % y: for fictional label oracle
  %  1: sampling
  %  2: argmax
  %  3: always 0
  %  4: always 1
  %  5: soft
  policy.name = 'sequential';
  policy_num  = int32(policy_num);
  
  policy_code = idivide(policy_num, 10);
  switch policy_code
    case {1,2}  % myopic
      policy.subname = 'myopic';
      policy.lookahead = policy_code;
    case 3
      policy.subname = 'ens';
    case 4
      policy.subname = 'ims';
  end
  
  label_code  = mod(policy_num, 10);
  label_oracles = {'sample', 'argmax', 'always0', 'always1', 'soft'};
  policy.fiction_oracle = label_oracles{label_code};
end
