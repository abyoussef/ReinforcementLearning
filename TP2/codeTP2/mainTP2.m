%% Build your own bandit problem 

% this is an example, please change the parameters or arms!
Arm1=armBernoulli(0.3);
Arm2=armBernoulli(0.25);
Arm3=armBernoulli(0.2);
Arm4=armBernoulli(0.1);

MAB={Arm1,Arm2,Arm3,Arm4};

% bandit : set of arms

NbArms=length(MAB);

Means=zeros(1,NbArms);
for i=1:NbArms
    Means(i)=MAB{i}.mean;
end

% Display the means of your bandit (to find the best)
Means
muMax=max(Means);


%% Comparison of the regret on one run of the bandit algorithm

T=5000; % horizon

[rew1,draws1]=UCB(T,MAB);
reg1=muMax*(1:T) - cumsum(rew1);
[rew2,draws2]=TS(T,MAB);
reg2=muMax*(1:T) - cumsum(rew2);


plot(1:T,reg1,1:T,reg2)


%% (Expected) regret curve for UCB and Thompson Sampling


