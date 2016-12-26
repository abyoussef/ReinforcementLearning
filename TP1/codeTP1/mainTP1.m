%% Problem specification
clear; 
% state space (including a sick state)
max_height = 5;
S = max_height+1;
sick_state = max_height+1;
init_state = 1;
% action space (a=1 keep; a=2 cut)
A = 2;
% discount parameter
gamma = 0.90; % 1 / (1+r) 
% maintenance and planting cost
maintenance_cost = 1.0;
% planting cost
planting_cost = 1.0;
% price for selling a tree per unit 
sell_price=1.0;


%% Parameters of the dynamics 

% probability of getting sick
sick_prob = 0.1;
% probability of growing: growth_proba(k,j) is the probability of growing by j when the current height is k
growth=zeros(max_height-1,max_height-1);
% for i = 1 : max_height-1
%     growth(max_height-i,1:i)=1/i;
% end
growth(1:(max_height-3),1:3)=1/3;
growth(max_height-2,1:2)=1/2;
growth(max_height-1,1)=1;


%% The MDP
% computes the transition array P=P(x,y,a) and expected value matrix R=R(x,a)
disp('Computing the true MDP...');
[P,R] = tree_MDP(max_height, A, sick_prob, growth,maintenance_cost, planting_cost,sell_price);


%% Parameters for the RL part

% number of trajectories used in MC or TD(0)
runs = 250;
% maximum number of steps per trajectory in MC and TD(0) (this will limit
% the accuracy of the two methods)
max_step = 1000;

%% Example of policy that may be evaluated
pi = ones(S,1);
% keep
%pi(1:10) = 1;
% cut
pi(ceil(max_height/2+1):max_height) = 2;
pi(sick_state) = 2;


%% Policy evaluation: DP (matrix inversion) versus RL (Monte-Carlo)

%RL (Monte-Carlo) in runs trajectories and max_step of steps in each
tMC = tic ;
Vmc = MCeval( pi,P,R,gamma,runs,max_step) ;
tMC = toc(tMC); 
% DP matrix 

tDP = tic ;
Vdp = DPeval(pi, P,R, gamma ); 
tDP = toc(tDP) ; 

%%
figure(); 

%% Policy Iteration
subplot(1,2,2);

[pf2,Vf] = polIter( pi,P,R,gamma ); 
%% Value Iteration
iter =10;

subplot(1,2,1); 
pf = valIter( pi,P,R,gamma,iter ,Vf ); 


