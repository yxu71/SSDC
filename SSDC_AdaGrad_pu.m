function [Obj,it,avgw]=SSDC_AdaGrad_pu(last, X_p, X_u, pi_p, lss, regs, eta0, T0, gamm, G,  w0, K, lambda, theta,...
                       epsilon, delta, dense,prox, idx_p, idx_u, idx_n,evalk, D)


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
disp(sprintf('T0 = %d | # of Gradients = %d | obj = %.15f', T0, 0, eps0));

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
    s_t = zeros(1,d);
    eta_t = eta0/sqrt(k);
    c_d_grad = zeros(1,d);
    %for tau=1:t
    tau = 1;
    while 1  
        i_p = idx_p(count);
        i_u = idx_u(count);
        i_n = idx_n(count);
        [g_p_grad, ind_p] = loss_grad(X_p(:,i_p), 1,  w,  lss, delta);
        [g_u_grad, ind_u] = loss_grad(X_u(:,i_u), -1, w,  lss, delta);
        [g_n_grad, ind_n] = loss_grad(X_p(:,i_n), -1, w0, lss, delta);
        d_grad = zeros(1,d);
        d_grad(ind_p) = pi_p * g_p_grad + d_grad(ind_p);
        d_grad(ind_u) = g_u_grad + d_grad(ind_u);
        d_grad(ind_n) = d_grad(ind_n) - pi_p*g_n_grad; 
        d_grad = d_grad + w * lambda;
        
        c_d_grad = c_d_grad + d_grad; 
        s_t = s_t + d_grad .* d_grad;
        H_t = 2 * G * ones(1,d) + sqrt(s_t);
        w = w0 - c_d_grad * eta_t./(gamm * eta_t * tau * ones(1,d) + H_t) ;
         
        if prox == 1
         w = map0(D, w);
        end
        
        if last == 0 
            if dense == 1 
                % Dense 
                avgw = avgw + w;
            else  
                % Sparse
                ind = find(w);
                avgw(ind) = avgw(ind) + w(ind);
            end
        end
        
        count=count+1;
        if count >= length(idx_p)
            break
        end
        
        maxH = 2*G + max(sqrt(s_t));
        if T0 > 1 
            if tau >=  T0 * k
                break;
            end
        else
            if tau >= 8 * maxH /(eta_t * gamm)
                break;
            end
        end
       tau = tau + 1;
    end

    
     T = T + 3*tau;
     Tcpu = Tcpu + (cputime-st2);
     Time = Time + toc(st1);
     
     if last == 0
        avgw = avgw/tau;
        w0 = avgw;
     end
     
     if last == 1
        w0 = w;
     end
     
    if mod(k, evalk)==0
        it(k/evalk,1) = Tcpu; % cpu time
        it(k/evalk,2) = Time; % running time
        it(k/evalk,3) = T; % iteraction 
        Obj(k) = loss_obj(X_p,X_u,pi_p,w0,lss,lambda,theta,delta);
        it(k/evalk,4) = Obj(k); 
        disp(sprintf('# of Gradients = %d | obj = %.15f | tau = %d | cpu =% d | time = %d | last = %d', ...
            T, Obj(k),tau,ceil(Tcpu),ceil(Time),  max(w0)));
        
        if T >= 3.1e7
            break
        end
    end

end
Obj = it;


