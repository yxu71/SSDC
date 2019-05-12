function [objres] = r_obj(w,d,regs,lambda,theta)

% regs
% 1: 'Clapped_l1'
% 2: 'SCAD'
% 3: 'MCP'
% 4: 'LSP'

 if regs == 1 %'Clapped_l1'
    objres = lambda * sum(min(abs(w), theta));
 end 
 
 if regs == 2 % 'SCAD'
    index2 = find(lambda < abs(w) & abs(w) <= theta* lambda);
    index1 = find(abs(w) <= lambda);
    w2 = w(index2);
    w1 = w(index1);
    d2 = length(index2);
    objres = (d - length(index1) - d2)*(theta+1)*lambda^2/2;
    objres = objres + lambda*sum(abs(w1)) - (norm(w2,2)^2 - (2*lambda*theta)*sum(abs(w2)) + d2 *lambda^2)/(2*(theta-1));
 end
 
 if regs == 3 % 'MCP'
    index1 = find(abs(w) <= lambda*theta);
    d2 = d - length(index1);
    w1 = w(index1);
    objres = lambda*sum(abs(w1)) - norm(w1)^2/(2*theta) + d2*theta*lambda^2/2;
 end

 if regs == 4 % 'LSP'
    objres = lambda*sum(log(1+abs(w)/theta));
 end
