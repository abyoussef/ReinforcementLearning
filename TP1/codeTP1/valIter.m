function [ pf ] = valIter( pi,P,R,gamma,iter ,Vf)
%VALITER do value iteration to find the optimal policy in iter iteration. 
    pf = zeros(size(pi,1),1); 
    V = DPeval(pi, P,R, gamma );  
    
    val = [V];
    for r = 1 : iter 
        V  = Vval( pi , V,P,R,gamma ) ;   
        val = [val , V ] ; 
    end
    
    
    for i = 1:size(pi,1) 
        if Qval( i,1, V,P,R,gamma ) > Qval( i,2, V,P,R,gamma )
           pf(i) = 1 ;  
        else 
           pf(i) = 2 ; 
        end
    end
     val = val - val(:,end);
     val = max(abs(val)); 
plot(val,'-r','Linewidth',2) ;
title(sprintf('Value iteration in i = %d iteration',iter)); 
end

