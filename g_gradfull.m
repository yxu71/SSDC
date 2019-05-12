function g = g_gradfull(X,y,w,lss,dlta)

% lss: loss function
  % 1: 'hinge'
  % 2: 'logistic' 
  % 3: 'least square'
  % 4: 'huber'
  % 5: 'squared hinge'

[d,n]=size(X);
  
    if lss == 2 %'logistic'
    	pred = -y.*(w*X); % 1*n
    	Exp = exp(pred); % 1*n
    	temp = Exp./(1+Exp); % 1*n
    	idx_NaN = find(Exp == inf);
    	temp(idx_NaN) = 1; 
    	g =(-y.*temp/n)*X'; % 1-by-d
    end
    if lss == 3 %'least'
        g =(w*X-y)*X'/n;
    end
    if lss == 4 %'huber'
       pred = w*X - y; %1*n
       pred(pred > dlta) = dlta;
       pred(pred < -dlta) = -dlta;
       g = pred*X'/n;		
    end
    if lss == 5 % 'squared hinge'
       ytemp = y;
	   hinge = 1 - (w*X).* ytemp; %1-by-n
 	   ytemp(hinge<0) = 0;
	   %g_bar = X*ytemp' /(-n); % hinge loss
	   g = (ytemp.*hinge/(-n/2))*X'; % 1-by-d squared hinge loss
    end

    
    if lss == 6 % non-linear least square loss with sigmod function
       temp = 1./(1+exp(-w*X));
       g =((temp-y).*temp.*(1-temp))*X'/n;
    end
   
    if lss == 7 % truncated least square  
             % dlta: tuncation parameter \alpha
    	temp = w*X-y;
    	g =(temp./(n+temp.^2./(dlta/n)))*X';
    end 
    
