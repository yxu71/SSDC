function [Obj,it,avgw]=SSDC_SPG(X, y, lss, regs, lambda, theta, dlta, ...
                        eta0, T0, gamma, w0, K, dense, idx)

[d,n]=size(X);
if eta0==0
    eta0 = 1./max(sum(X.*X,1));
end
if isempty(w0)
    w0=zeros(1,d);
end

eps0=g_obj(X,y,lss,w0,dlta) + r_obj(w0,d,regs,lambda,theta);
disp(sprintf('epoch=%d, obj=%.15f', 0, eps0));
Obj=[];
it=[];
T = 0;
Time = 0;
Tcpu = 0;
count = 1;

for k=1:K
    st1 = tic;
    st2 = cputime;
    w = w0;  
    avgw = zeros(1,d);
    t = T0 * k;    
    gradH = h_grad(w0,d,regs,lambda,theta);
    v = gradH + gamma*w0;
    
    for tau=1:t
        if count >= length(idx)
            break
        end
        eta = eta0/(1+tau);
        i = idx(count);
        count=count+1; 
        
        [gradG, ind] = g_grad(X(:,i),y(i),w,lss,dlta); 
        w = (v + w./eta)./(gamma+1/eta);
        w(ind) = w(ind) - gradG./(gamma+1/eta);
        w = l1_soft(w,lambda/(gamma+1/eta));
        
        if dense == 1 
        % Dense 
           avgw = avgw + (tau + 1) * w;
        else  
        % Sparse
           ind = find(w);
           avgw(ind) = avgw(ind) + (tau + 1)*w(ind);
        end
    end

    Tcpu = Tcpu + (cputime-st2);
    it(k,1) = Tcpu; % cpu time
    Time = Time + toc(st1);
    it(k,2) = Time; % running time
    T = T + t;
    it(k,3) = T; % iteraction 
    avgw=avgw./(t*(t+3)/2);
    w0 = avgw; 
    Obj(k) = g_obj(X,y,lss,w0,dlta) + r_obj(w0,d,regs,lambda,theta);
      disp(sprintf('epoch=%d, obj=%.15f, T=%d,cpu=%d, time=%d', ...
            k, Obj(k),ceil(T),ceil(Tcpu),ceil(Time)));
    if T >= length(idx)
        break
    end
        
end


it = [0,0,0; it];
Obj = [eps0, Obj];
