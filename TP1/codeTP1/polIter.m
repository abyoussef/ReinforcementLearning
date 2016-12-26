function [ pf ,Vf] = polIter( pi,P,R,gamma )
%POLITER Policy iteration to find the optimal policy
V = DPeval(pi, P,R, gamma );  
Vf = zeros(size(V,1),1); 
pf = ones(size(pi,1),1) ;
eps = 1 ;   
iter=0 ; 
val = [V] ; 
while max(abs(V - Vf)) > eps 
   iter = iter+1 ; 
   V = Vf ; 
   for i = 1:size(pi,1) 
        if Qval( i,1, V,P,R,gamma ) > Qval( i,2, V,P,R,gamma )
           pf(i) = 1 ;  
        else 
           pf(i) = 2 ; 
        end
   end
   Vf = DPeval(pf,P,R,gamma) ; 
   val = [val , V]; 
end

val = val - val(:,end);
val = max(abs(val)); 
size(val)
plot(val,'-b','Linewidth',2) ;
title(sprintf('Policy iteration in i = %d iteration',iter)); 

end

