function [g, ind] = loss_grad(X,y,w,lss,delta)
% d: dimension of feature
% X: d-by-1
% y: 1-by-1
% w: 1-by-s


ind = find(X);
g=0;

if lss == 1 % 'hinge'
   x = X(ind);
   temp = y*x;
   pred = w(ind)*temp;
   if pred < 1
     g = -temp';
   end
end

if lss == 2 % 'logistic' 
   x = X(ind);
   temp = -y*x;
   pred = w(ind)*temp;
   if pred > 37
      g = temp';
   elseif pred > -746
      Exp = exp(pred);
      g = (Exp/(1+Exp))*temp';
   end
end

if lss == 3 % 'least'
    x = X(ind);
    pred = w(ind)*x - y;
    g = pred*x';
end

if lss == 4  % 'huber'
   x = X(ind);
   pred = w(ind)*x - y;
   if abs(pred) <= delta
      g = pred*x'; 
   else
      if pred > 0
        g = delta*x';
      elseif pred < 0
        g = -delta*x';
      end
   end
end

if lss == 5 % 'squared hinge'
   x = X(ind);
   temp = y*x;
   pred = w(ind)*temp;
   if pred < 1
     g =2*(pred-1)*temp';
   end
end

if lss == 6 % 'hinge'
    x = X(ind);
    temp = y*x; 
    pred = w(ind)*temp;
    if pred < 1
      g = -temp';
    end
end

if lss == 7 % 'absolute'
    x = X(ind);
    pred = w(ind)*x - y;   
    [row, col] = size(x);
    if pred > 0
        g = x';
    elseif pred == 0
        g = 0;
    else
        g = -x';
    end
end








