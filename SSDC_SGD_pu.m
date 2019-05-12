function [Obj,it,avgw]=SSDC_SGD_pu(last, X_p, X_u, pi_p, lss, regs, eta0, T0, gamm, w0, K, lambda, theta,...
                       epsilon, delta, dense,prox, idx_p, idx_u, idx_n, evalk, D)


[d,n] =size(X_p);
if eta0==0
    eta0 = 1./max(sum(X.*X,1));
end
if isempty(w0)
    w0=zeros(1,d);
end

if epsilon<1
    K = max(1/eta0, 1)/epsilon^4;
end

avgw = w0;
eps0=loss_obj(X_p,X_u,pi_p,avgw,lss,lambda,theta,delta);
disp(sprintf('epoch=%d, obj=%.15f', 0, eps0));
Obj=[];
it=[];
T = 0;
Time = 0;
Tcpu = 0;
count = 1;

it(1,1) = 0;
it(1,2) = 0;
it(1,3) = 0;
it(1,4) = eps0;

for k=1:K
    st1 = tic;
    st2 = cputime;
    w = w0;
    weights = 0;
    avgw = zeros(1,d);
    t = T0 * k;
    for tau=1:t
        if count >= length(idx_p)
            break
        end
        eta_t = eta0/(1+tau);
        i_p = idx_p(count);
        i_u = idx_u(count);
        i_n = idx_n(count);
        count=count+1;        
        [g_p_grad, ind_p] = loss_grad(X_p(:,i_p), 1, w,lss,delta);
        [g_u_grad, ind_u] = loss_grad(X_u(:,i_u),-1,w,lss,delta);
        [g_n_grad, ind_n] = loss_grad(X_p(:,i_n), -1, w0,lss,delta);
        d_grad = zeros(1,d);
        d_grad(ind_p) = pi_p * g_p_grad + d_grad(ind_p);
        d_grad(ind_u) = g_u_grad + d_grad(ind_u);
        d_grad(ind_n) = d_grad(ind_n) - pi_p*g_n_grad;
        d_grad = d_grad + lambda * w;
        
        w = (w + gamm * eta_t * w0 - d_grad * eta_t)/(gamm * eta_t + 1);
        
         if(prox == 1)
            w = map0(D, w);
         end

        if last == 0 
            if dense == 1 
                % Dense 
                avgw = avgw + (tau + 1) * w;
            else  
                 % Sparse
                ind = find(w);
                avgw(ind) = avgw(ind) + (tau + 1)*w(ind);
            end
          weights = weights + (tau + 1);
        end  
    end
    
    Tcpu = Tcpu + (cputime-st2);
    Time = Time + toc(st1);
    T = T + 3*t;
    
    if last == 0
        if(tau ~= t)
            avgw = w0;
        else
            avgw=avgw/weights;
        end
        
        w0 = avgw; 
    end
    
    if last == 1
        w0 = w;
    end
        
    
    
    
    
    if(mod(k, evalk) == 0)
        it(k/evalk + 1,1) = Tcpu; % cpu time
        it(k/evalk + 1,2) = Time; % running time
        it(k/evalk + 1,3) = T; % iteraction 
        Obj(k) = loss_obj(X_p,X_u,pi_p,w0,lss,lambda,theta,delta);
        it(k/evalk + 1,4) =Obj(k);
        disp(sprintf('# of Gradients = %d | obj = %.15f| T = %d | cpu =%d | time=%d | eta0 = %d | last = %d', ...
            T, Obj(k),ceil(T),ceil(Tcpu),ceil(Time), eta0, last));
    end

    
    if T >= (3*length(idx_p))
        break
    end
        
end
Obj = it;


