function [ a ] = epsGreedy(Q,eps )
%EPSGREEDY for choosing the action with epsilon greedy policy
    [S,A] = size(Q);
    p  = binornd(1,eps,[S,1]) ; 
    [~,indexMax] = max(Q,[],2) ; 
    a = randi(A,[S,1]) .* p + indexMax .* (1-p) ; 
end

