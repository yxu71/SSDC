function [objres] = g_obj(X,y,lss,w,dlta)

% lss: loss function
% 1: 'hinge'
% 2: 'logistic'
% 3: 'least square'
% 4: 'huber'
% 5: 'squared hinge'

 pred=w*X;

 if lss == 1 %'hinge'
    objres = max(0, 1 - pred.* y);
 end 

 if lss == 2 % 'logistic'
    objres = -y.*pred;
    id = find(objres <= 709);
    objres(id) = log(1+exp(objres(id)));
 end
 
 if lss == 3 % 'least square'
    objres =  0.5.*(pred - y).^2;
 end

 if lss == 4 % 'huber'
    objres = pred - y;
    id1 = find(abs(objres)<=dlta);
    id2 = find(abs(objres)>dlta);
    objres(id1) = 0.5*objres(id1).^2;
    objres(id2) = dlta*(abs(objres(id2)) - 0.5*dlta);
 end

 if lss == 5 % 'squared hinge'
    objres = max(0, 1 - pred.* y);
    objres = objres.^2;	
 end

 if lss == 6 % non-linear least square loss with sigmod function
    pred = 1./(1+exp(-pred));
    objres = (pred - y).^2;
 end
        
 if lss == 7 % truncated least square  
	     % dlta: tuncation parameter \alpha
    objres =  (0.5*dlta).*log(1+(pred - y).^2./dlta);
 end

 %reg = lambda*sum(abs(w));
 objres = mean(objres); %+ reg;

 
 
 
 
