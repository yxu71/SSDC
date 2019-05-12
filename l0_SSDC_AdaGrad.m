function [Obj,it,avgw]=l0_SSDC_AdaGrad(X, y, lss, lambda, dlta, eta0, gamma, mu, K, T0, w0, G, dense,idx, maxT)
                                    
%dlta is for hinge loss
%theta for non-covex regularizer


[d,n]=size(X)
if eta0==0
    eta0 = 1./max(sum(X.*X,1));
end
if isempty(w0)
    w0=zeros(1,d);
end

avgw = w0;
eps0=g_obj(X,y,lss,w0,dlta) + lambda*nnz(w0);
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
    avgw = w0;
    eta = eta0/sqrt(k);
 
    gradH = mu.*w0;   
    gt_sum = 0;
    st_squre = 0;
    
    tau = 1;
    while 1
        if count > length(idx)
            break
        end
        i = idx(count);
        count=count+1; 
        [gradG, ind] = g_grad(X(:,i),y(i),w,lss,dlta); 
        gt = -gradH + mu.*w;
        gt(ind) = gt(ind) + gradG; % compute g_t
        st_squre = st_squre + gt.^2; % compute s_t^2
        gt_sum = gt_sum + gt; % compute sum_{tau=1}^t gt
        
        Ht = sqrt(st_squre);
        maxHt = max(Ht);
        Ht = 2*G + Ht; % Ht is a vector here, while it is a diaginal matrix in the paper.
        gt_avg = gt_sum./tau; %  averaged gt: will be used in the update of x_{\tau+1}.
        
        prm = gamma + Ht./(tau*eta); % vector
        w = w0 - gt_avg./prm;
        w(abs(w)<=sqrt(2*lambda./prm)) = 0; % l0 mapping: hard-thresholding           
        
	if dense == 1 
        % Dense 
           avgw = avgw + w;
        else  
        % Sparse
           ind = find(w);
           avgw(ind) = avgw(ind) + w(ind);
        end

	if T0 >= 1  
	   if tau >= k*T0 % practical 
	      break;
	   end
	else
                M = 4*(2*G+maxHt)/(gamma*eta);
	   if tau > M 
              break;
	   end
        end
        tau = tau + 1;
    end

    Tcpu = Tcpu + (cputime-st2);
    it(k,1) = Tcpu; % cpu time
    Time = Time + toc(st1);
    it(k,2) = Time; % running time
    T = T + tau;
    it(k,3) = T; % iteraction 
    avgw=avgw/tau;
    w0 = avgw; 
    Obj(k) = g_obj(X,y,lss,w0,dlta) + lambda*nnz(w0);
      disp(sprintf('epoch=%d, obj=%.15f, T=%d,cpu=%d, time=%d', ...
            k, Obj(k),ceil(T),ceil(Tcpu),ceil(Time)));    
    if T > length(idx) | T > maxT
        break
    end
        
end


it = [0,0,0; it];
Obj = [eps0, Obj];
