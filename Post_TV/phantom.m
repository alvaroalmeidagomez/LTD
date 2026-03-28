clc;
clear;
close all;

%% ================= PROJECT PRESENTATION =================
disp('======================================================================');
disp('#### Project supported by Centro de Modelamiento Matematico (Chile) ####');
disp('#### Angular Regularization for Sparse-Angle Tomographic Projections ####');
disp('#### Sparse Tomographic Reconstruction of the Phantom               ####');
disp('======================================================================');
fprintf('\n');

%% ================= INPUT PARAMETERS =================
num_train_proj = input('Enter the number of training projections: ');
num_learn_proj = input('Enter the number of projections to be learned: ');
fprintf('\n');

%% ================= DATA GENERATION =================
disp('Generating data and training projections...');

filename = 'phantom.jpg';

% Read image and ensure it is 256x256 grayscale double
P = imread(filename);
if size(P, 3) > 1
    P = rgb2gray(P); 
end
P = imresize(im2double(P), [256 256]);

% Store original for error calculation and comparison
P_original = P; 
img_size = size(P, 1); 

% Add uniform noise to the image
% Note: rand(size(P)) is safer and better practice than rand(256)
P = P + 0.2 * rand(size(P)); 

% Generate random training angles and compute the Radon transform (sinogram)
theta_train = sort(180 * rand(1, num_train_proj));
[R_train, xp] = radon(P, theta_train);

%% ================= PROPOSED METHODOLOGY =================
disp('Executing Proposed Methodology (Angular Regularization)...');
t_prop = tic;

% Generate random target angles to learn/interpolate
theta_learn = sort(180 * rand(1, num_learn_proj));
R_learn = zeros(length(xp), length(theta_learn));

for i = 1:num_learn_proj
    theta_target = theta_learn(i);
    [epsilon, idx] = min(abs(theta_train - theta_target));
    
    if epsilon == 0
        R_learn(:, i) = R_train(:, idx);
    else
        % Vectorized weight calculation for maximum performance
        distances = abs(theta_target - theta_train);
        weights = exp(-(distances / epsilon).^2);
        
        % Interpolate the projection based on Gaussian weights
        R_learn(:, i) = (R_train * weights') / sum(weights);
    end
end
time_prop = toc(t_prop);
fprintf('Methodology execution time: %.2f seconds\n\n', time_prop);

%% ================= IMAGE RECONSTRUCTION =================
disp('Reconstructing images using inverse Radon transform (FBP)...');

% 1. Reconstruct using the learned/interpolated sinogram (Proposed)
rec_proposed = iradon(R_learn, theta_learn, 'linear', 'Ram-Lak', 1, img_size);

% 2. Reconstruct using only the sparse training sinogram (Standard FBP)
rec_fbp = iradon(R_train, theta_train, 'linear', 'Ram-Lak', 1, img_size);


disp('Applying Total Variation (TV) Regularization...');

% 3. TV Reconstruction using the proposed methodology as the initial guess
[X_tv_proposed, time_vec_prop, error_vec_prop] = Total_variation_CT(R_train, theta_train, rec_proposed, P_original);

% 4. TV Reconstruction using standard FBP as the initial guess
[X_tv_fbp, time_vec_fbp, error_vec_fbp] = Total_variation_CT(R_train, theta_train, rec_fbp, P_original);


%% ================= PLOTTING RESULTS =================

% --- Figure 1: Image Comparison (2x2 Grid) ---
% Using subplot to display all images in a single window for easy comparison
figure('Name', 'Tomographic Reconstruction Comparison', 'Position', [100, 100, 900, 900]);

% Top-Left: Original Image
subplot(2, 2, 1);
imshow(P_original);
title('Original Ground Truth Image');

% Top-Right: Proposed Methodology
subplot(2, 2, 2);
imshow(rec_proposed);
title('Proposed Methodology');

% Bottom-Left: TV from standard FBP
subplot(2, 2, 3);
imshow(X_tv_fbp);
title('TV Regularization (from standard FBP)');

% Bottom-Right: TV from Proposed Methodology
subplot(2, 2, 4);
imshow(X_tv_proposed);
title('TV Regularization (from Proposed Methodology)');


% --- Figure 2: Performance Tracking ---
figure('Name', 'Performance: Time vs Error');
plot(time_vec_prop, error_vec_prop, 'LineWidth', 1.5);
hold on;
plot(time_vec_fbp, error_vec_fbp, 'LineWidth', 1.5);
grid on;

title('Convergence: Execution Time vs Reconstruction Error');
xlabel('Time (seconds)');
ylabel('Frobenius Norm Error');
legend({'TV with Proposed Init', 'TV with Standard FBP Init'}, 'Location', 'best');

clc
disp('Process Complete.');

disp('======================================================================');
disp('#### Project supported by Centro de Modelamiento Matematico (Chile) ####');
disp('======================================================================');