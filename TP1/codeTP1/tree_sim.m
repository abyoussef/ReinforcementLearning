function [ next, reward ] = tree_sim( state, action , P ,R  )
%TREE_SIM receives as input a state and an action and it returns
% the next state and the reward.
    
    next = simu(P(state,:,action));
    reward = R(state,action);

end

