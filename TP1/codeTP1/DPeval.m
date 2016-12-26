function [ V ] = DPeval(pol, P,R, gamma )
%DPeval Policy evalution with Dynamic Programming (Matrix inversion) 
% pol is the policy and P is the dynamic and R is the reward 
    n = size(pol,1) ; 
    Rpol = zeros(n,1);
    Ppol = zeros(n,n); 
    
    % Define the Ppol and Rpol
    for i = 1:n 
       Rpol(i) = R(i,pol(i)); 
       Ppol(i,:) = P(i,:,pol(i)); 
    end
    
    V = inv( eye(n) - gamma * Ppol) * Rpol ;
    

end

