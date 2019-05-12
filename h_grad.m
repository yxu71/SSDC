function [g] = h_grad(w,d,regs,lambda,theta)

% g: gradient of function r2 
% d: dimension of feature
% w: 1-by-d

% regs
  % 1: 'r2 of Clapped_1'
  % 2: 'r2 of SCAD' 
  % 3: 'r2 of MCP'
  % 4: 'r2 of LSP'

if regs == 1 % 'r2 of Clapped_l1'
   g = zeros(1,d);
   g(w>theta) = lambda;
   g(w<-theta) = -lambda;
end

if regs == 2 % 'r2 of SCAD'
   g = zeros(1,d);
   index2 = find(lambda < abs(w) & abs(w) <= theta* lambda);
   index3 = find(abs(w) > theta* lambda);
   w2 = w(index2);
   g(index2) = (w2 - lambda .* sign(w2)) ./ (theta - 1);
   g(index3) = lambda * sign(w(index3));
end

if regs == 3 % 'r2 of MCP'
   g = lambda.*sign(w);
   index = find(abs(w) <= lambda*theta);
   g(index) = w(index)./theta;
end

if regs == 4  % 'r2 of LSP'
  g = sign(w).*(lambda - lambda./(theta + abs(w)));
end



