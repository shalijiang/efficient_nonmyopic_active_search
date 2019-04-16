% This file defines the codes of the policies supported.
% All policies are hard-coded with numbers; there is not necessarily a
% meaningful rule of how the numbers are decided, since the policies were
% gradually added, so we just kept using different numbers.
% This allows easy parameterization of the policies using different parts
% of the numbers.
% The names of the constants defined here match the names in the paper.

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
