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


%% Example of policy that may be evaluated
pi = zeros(S,1);

%% Q-Learning 

eps = 0.4
% Initialization Q
Q = ones(S,A) ; 
%Initialization of action 
a = epsGreedy(Q,eps); 
% Number of visits N
N = ones(S,A) ; 

next = zeros(S,1);
r    = zeros(S,1); 
%Number of steps
T = 1000 ; 

%% Q-Learning Iterations
for i = 1 : T
    % update number of visit
    for k = 1:A
       N(:,k) = ones(S,1) .* (a==k) + N(:,k);
    end
    alpha = 1./diag(N(:,a)); 
    for j = 1:S 
        [next(j),r(j)] = tree_sim(j, a(j) , P ,R  ); 
        Q(j,a(j)) = (1 - alpha(j)) * Q(j,a(j)) + alpha(j) ...
            * (r(j) + gamma * max(Q(next(j),:))) ; 
    end
    [V(:,i),pi(:,i)] =   max(Q,[],2);
    a = epsGreedy(Q,eps); 
    
end