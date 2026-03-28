function [x, time_vec, error_vec] = Total_variation_CT(sinogram, theta, x, P_original)
% TOTAL_VARIATION_CT Iterative CT reconstruction with Total Variation (TV) regularization.
%
% This function solves the optimization problem using a Forward-Backward 
% splitting method (Gradient Descent + Proximal TV step).
%
% Inputs:
%   sinogram   - Projection matrix (detectors x angles).
%   theta      - Projection angles vector (in degrees).
%   x          - Initial estimate of the reconstructed image.
%   P_original - Ground truth image (used for calculating iteration error).
%
% Outputs:
%   x          - Final reconstructed image.
%   time_vec   - Vector tracking cumulative execution time per iteration.
%   error_vec  - Vector tracking the Frobenius norm error per iteration.

    %% 1. Initial Setup and Parameter Definition
    
    % Determine image size (N) based on the sinogram dimensions.
    % The iradon function assumes the image is slightly smaller than the sinogram.
    [num_detectors, ~] = size(sinogram);
    N = 2 * floor(num_detectors / (2 * sqrt(2))) - 2; % Safe estimation of N
    
    % Optimization parameters
    lambda    = 0.5;   % TV regularization parameter
    num_iter  = 500;   % Number of iterations
    step_size = 0.001; % Gradient descent step size
    
    % Initialize tracking arrays to preallocate memory
    error_vec = zeros(num_iter, 1);
    time_vec  = zeros(num_iter, 1);
    
    fprintf('Starting TV reconstruction for image of size %dx%d...\n', N, N);
    
    % Start the execution timer
    t_start = tic;

    %% 2. Optimization Loop
    for i = 1:num_iter
        
        % --- A. GRADIENT STEP (Data Fidelity) ---
        
        % 1. Forward project the current image estimate
        Ax = radon(x, theta);
        
        % 2. Dimension Adjustment
        % The radon function often returns a matrix larger than the original sinogram.
        % We crop Ax so it exactly matches the input sinogram dimensions.
        if size(Ax, 1) ~= num_detectors
            diff_size = size(Ax, 1) - num_detectors;
            idx_start = floor(diff_size / 2) + 1;
            idx_end   = idx_start + num_detectors - 1;
            Ax = Ax(idx_start:idx_end, :);
        end
        
        % 3. Compute the error in the projection (sinogram) domain
        error_sinogram = Ax - sinogram;
        
        % 4. Backproject the error to compute the gradient: A'(Ax - b)
        % We use 'none' to apply a pure backprojection operator without FBP filtering.
        grad = iradon(error_sinogram, theta, 'linear', 'none', 1, N);
        
        % 5. Update the image via Gradient Descent
        x = x - step_size * grad;
        
        % --- B. PROXIMAL STEP (TV Denoising) ---
        
        % Apply Chambolle's algorithm for edge-preserving smoothing.
        try
            x = tv_denoise_chambolle(x, lambda * step_size);
        catch
            warning('tv_denoise_chambolle function not found. Terminating TV step early.');
            break;
        end
        
        % --- C. PERFORMANCE TRACKING ---
        
        % Record elapsed time
        time_vec(i) = toc(t_start);
        
        % Compute the Frobenius norm error compared to the original image.
        % Note: Single quotes ('fro') are safer for MATLAB 2017b compatibility.
        error_vec(i) = norm(x - P_original, 'fro'); 
        
    end
    
    % Optional: Notify the user when complete
    fprintf('Reconstruction completed in %.2f seconds.\n', time_vec(end));

end