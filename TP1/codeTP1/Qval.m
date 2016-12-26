function [ v ] = Qval( i,a, V,P,R,gamma )
%Qval Compute the Q-Value for state i and action a 
    v = R(i,a) + gamma * P(i,:,a) * V ;
end

