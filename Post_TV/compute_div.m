function d = compute_div(p)
    % Divergence (adjoint of the gradient)
    px = p(:,:,1);
    py = p(:,:,2);
    
    dx = px(:, [1, 1:end-1]) - px;
    dx(:, 1) = px(:, 1);
    dx(:, end) = -px(:, end-1);
    
    dy = py([1, 1:end-1], :) - py;
    dy(1, :) = py(1, :);
    dy(end, :) = -py(end-1, :);
    
    d = dx + dy;
end