function [objres] = h_obj(w,regs,lambda,theta)

% regs
% 1: 'r2 of Clapped_l1'
% 2: 'r2 of SCAD'
% 3: 'r2 of MCP'
% 4: 'r2 of LSP'

 if regs == 1 %'r2 of Clapped_l1'
    objres = lambda * sum(max(abs(w) - theta, 0));
 end 

 if regs == 2 % 'r2 of SCAD'
    w2 = w(lambda < abs(w) & abs(w) <= theta* lambda);
    w3 = w(abs(w) > theta* lambda);
    objres = sum((w2.*w2 - (2*lambda).*abs(w2) + lambda^2)./(2*(theta-1)));    
    objres = objres + lambda * sum(abs(w3) - (theta+1)*lambda/2);
 end
 
 if regs == 3 % 'r2 of MCP'
    index1 = find(abs(w) <= lambda*theta);
    index2 = logical(ones(1,d) - index1);
    objres = norm(w(index1))^2/(2*theta) + lambda*sum(abs(w(index2))-theta*lambda/2);
 end

 if regs == 4 % 'r2 of LSP'
    absW = abs(w);
    objres = lambda*sum(absW - log(1+absW./theta));
 end

 
 
 