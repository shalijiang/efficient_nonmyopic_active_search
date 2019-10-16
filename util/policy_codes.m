% This file defines the codes of the policies supported.
% All policies are hard-coded with numbers; there is not necessarily a
% meaningful rule of how the numbers are decided, since the policies were
% gradually added, so we just kept using different numbers.
% This allows easy parameterization of the policies using different parts
% of the numbers.
% The names of the constants defined here match the names in the paper.
%
% see util/get_policy_struct.m for the correspondance

%% The sequential policies in Jiang et al. (2017) paper (see Section 5) 
% are coded as follows (run with batch size 1):
ONE_STEP = 1;  % one-step 
TWO_STEP = 23; % two-step 
ENS      = 2;  % ENS 
RG       = 3;  
IMS      = 43; 

%% The batch policies in Jiang et al. (2018) (see Section 5) 
GREEDY     = 1;  % same as one-step
BATCH_ENS  = 2;  % share the same code with ENS


%% Sequential simulation policies
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
SS_ONE_S = 11;
SS_ONE_M = 12;
SS_ONE_0 = 13;
SS_ONE_1 = 14;

SS_TWO_S = 21;
SS_TWO_M = 22;
SS_TWO_0 = 23;
SS_TWO_1 = 24;

SS_ENS_S = 31;
SS_ENS_M = 32;
SS_ENS_0 = 33;
SS_ENS_1 = 34;

%% cost effective active search policies (NeurIPS 2019) 
% a few examples 
CENS20 = 5.102;
CENS30 = 5.103; 
CENS0_1 = 5.21;
CENS0_2 = 5.22;

%% artifically set the budget for ENS for cost effective setting
% these are just a few examples 
% proportional to remaining budget 
ENS0_3 = 2.3;  % lookahead for 30% of remaining budget 
ENS0_5 = 2.5;  % lookahead for 50% of remaining budget 
% constant 
ENS10 = 4.01;  % lookahead 10 steps 
ENS20 = 4.02;  % lookahead 20 steps
