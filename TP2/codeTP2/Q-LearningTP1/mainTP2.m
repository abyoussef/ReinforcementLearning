%% Problem specification
clear; 
% state space (including a sick state)
max_height = 20;
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
for i = 1 : max_height-1
    growth(max_height-i,1:i)=1/i;
end
% growth(1:(max_height-3),1:3)=1/3;
% growth(max_height-2,1:2)=1/2;
% growth(max_height-1,1)=1;


%% The MDP
% computes the transition array P=P(x,y,a) and expected value matrix R=R(x,a)
disp('Computing the true MDP...');
[P,R] = tree_MDP(max_height, A, sick_prob, growth,maintenance_cost, planting_cost,sell_price);


%% Q-Learning 
state = 1; 
eps = 0.4;
%Number of steps
T = 1000 ; 
% Initialization Q
Q = R ; 
% Value function and policy 
V = zeros(S,T);
pi = zeros(S,T) ; 
[V(:,1),pi(:,1)] = max(Q,[],2);

%Initialization of action 
a = epsGreedyAction(Q,eps,state); 
% Number of visits N
N = ones(S,A) ; 

%% Q-Learning Iterations
for i = 2 : T
    % update number of visit
    N(state,a) = N(state,a) + 1 ; 
    alpha = 1 / N(state,a);
    [next,reward] = tree_sim(state, a , P ,R  ); 
    Q(state,a) = (1 - alpha) * Q(state,a) + alpha ...
            * (R(state,a) + gamma * max(Q(next,:))) ; 
    
    [V(:,i),pi(:,i)] =   max(Q,[],2);
    state = next ; 
    a = epsGreedyAction(Q,eps,state); 
    
end

figure(1);
plot(abs(V(1,:)-V(1,end)),'-b','Linewidth',2) ;
title(sprintf('Performance w.r.t initial state in T = %d episodes',T)); 
figure(2);
plot(max(abs(V - V(:,end)),[],1),'-b','Linewidth',2) ;
title(sprintf('Performance in all states in T = %d episodes',T)); 
