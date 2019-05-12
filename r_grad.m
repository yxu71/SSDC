function [g] = r_grad(w,d,regs,lambda,theta)

% g: gradient of function r = r1 - r2 
% d: dimension of feature
% w: 1-by-d

% regs
  % 1: 'Clapped_1'
  % 2: 'SCAD' 
  % 3: 'MCP'
  % 4: 'LSP'
  


if regs == 1 % 'Clapped_l1'
   g = zeros(1,d);
   g(w>0 & w<theta) = lambda;
   g(w<0 & w>-theta) = -lambda;
end

if regs == 2 % 'SCAD' 
   g = zeros(1,d);
   index2 = find(lambda < abs(w) & abs(w) <= theta* lambda);
   index1 = find(abs(w) <= lambda);
   w2 = w(index2);
   g(index2) = (- w2 + (theta*lambda).*sign(w2))./(theta - 1);
   g(index1) = lambda .* sign(w(index1));
end

if regs == 3 % 'MCP'
   g = zeros(1,d);
   index = find(abs(w) <= lambda*theta);
   w1 = w(index);
   g(index) = lambda * sign(w1) - w1./theta;
end

if regs == 4  % 'LSP'
   g = lambda.* sign(w)./(theta + abs(w));
end



