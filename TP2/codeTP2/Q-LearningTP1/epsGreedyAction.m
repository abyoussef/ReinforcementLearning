function [ a ] = epsGreedyAction( Q,eps,s )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    A = size(Q,2) ; 
    p  = binornd(1,eps) ;
    [~,indexMax] = max(Q(s,:),[],2);
    a = randi(A) * p + indexMax * (1-p); 
end

