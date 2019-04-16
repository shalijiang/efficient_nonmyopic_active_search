function policy = get_policy_struct(policy_num)
% create a policy struct based on an integer policy_num

if policy_num == 1
  policy.name = 'greedy';
elseif policy_num == 2
  policy.name = 'batch-ens';
elseif policy_num == 3
  policy.name = 'random-greedy';
elseif policy_num > 6 && policy_num < 7  % used for rebuttal
  policy.transition_ratio = policy_num - 6;
  policy.name = 'uncertainty-greedy';
elseif policy_num >= 7 && policy_num < 8
  policy.epsilon = policy_num - 7;
  policy.name = 'uncertainty-greedy-mixed';
else  % sequential simulation policies
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
  %  5: soft (directly use the probabilities as labels)
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
