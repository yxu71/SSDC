function [objres] = loss_obj(X_p,X_u,pi_p,w,lss,lambda,theta,delta)
% lss = 4 sqaure loss.
g_objres = pi_p * g_obj(X_p, +1, lss, lambda, w, delta) + g_obj(X_u, -1, lss, lambda, w, delta);
h_objres = pi_p * g_obj(X_p, -1, lss, lambda, w, theta);
objres = g_objres - h_objres + lambda/2 * sum(w.*w);

if h_objres == Inf
    fprintf(sprintf('%d\n', Inf)); 
    fprintf(sprintf('%d\n', isnan(objres))); 
end

if objres == Inf || isnan(objres)
    fprintf('Something is wrong? \n');
end
    
    

