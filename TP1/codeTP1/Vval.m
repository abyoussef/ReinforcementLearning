function [ Vf ] = Vval( pi , Vi,P,R,gamma )
%Vval Compute Vf given a policy pi and Vf with Bellman equation 
    Vf = zeros(size(Vi,1),1) ; 
    for i = 1:size(Vi,1) 
        Vf(i) = Qval( i,pi(i), Vi,P,R,gamma ) ; 
    end
end

