function g = compute_grad(f)
    % Forward finite differences
    [rows, cols] = size(f);
    g = zeros(rows, cols, 2);
    g(:, 1:cols-1, 1) = diff(f, 1, 2);
    g(1:rows-1, :, 2) = diff(f, 1, 1);
end