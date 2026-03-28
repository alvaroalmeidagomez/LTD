function u = tv_denoise_chambolle(f, lambda)
    % Chambolle's algorithm for the TV proximal operator
    % It is efficient and only uses finite differences in MATLAB.
    [rows, cols] = size(f);
    p = zeros(rows, cols, 2); % Dual field
    tau = 0.24; % Time step for stability
    
    for i = 1:10 % A few inner iterations are enough
        % Gradient of the divergence
        div_p = compute_div(p);
        w = f - div_p;
        grad_w = compute_grad(w);
        
        % Update of the dual field
        norm_grad_w = sqrt(sum(grad_w.^2, 3));
        p = (p + tau * grad_w) ./ (1 + tau/lambda * norm_grad_w);
    end
    u = f - compute_div(p);
end