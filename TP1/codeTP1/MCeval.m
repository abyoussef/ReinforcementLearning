function [ V ] = MCeval( pi,P,R,gamma,N,T )
%MCeval Policy evaluation with N trajectories and T steps

n = size(pi,1); 
Gamma = gamma * ones(T,1); 
Gamma = [1 ; Gamma]; 
Gamma = cumprod(Gamma); 

reward = zeros(1,T+1);
V = zeros(n,1); 


for i = 1:n 
   for m = 1:N
       s = i ; 
       for t= 1:T+1 
            [n,r] = tree_sim( s, pi(s), P,R) ;
            reward(t) = r ; 
            s = n;
       end
       V(i) = V(i)  + reward * Gamma ; 
   end
   V(i) = V(i) / N ; 
end



end

