clc;
clear;
close all;

%% ================= PROJECT PRESENTATION =================
disp('======================================================================');
disp('#### Project supported by Centro de Modelamiento Matematico (Chile) ####');
disp('#### Angular Regularization for Sparse-Angle Tomographic Projections ####');
disp('#### Sparse Tomographic Reconstruction of the Abdomen               ####');
disp('======================================================================');
fprintf('\n');

%% ================= INPUT PARAMETERS =================
numTrainProj = input('Enter the number of training projections: ');
numLearnProj = input('Enter the number of projections to be learned: ');
fprintf('\n');

%% ================= DATA GENERATION =================
disp('Generating data and training projections...');

filename_char = 'Abdomen.png';
% Read image and ensure it is 256x256 grayscale
P = imread(filename_char);
if size(P,3) > 1, P = rgb2gray(P); end
P = imresize(im2double(P), [256 256]);
sz = size(P, 1); % Store original size (256)
P = P + 0.2 * rand(256);

theta_train = sort(180 * rand(1, numTrainProj));
[R_train, xp] = radon(P, theta_train);

%% ================= PROPOSED METHODOLOGY =================
disp('1/5 Executing Proposed Methodology (Angular Regularization)...');
t_prop = tic;
theta_learn = sort(180 * rand(1, numLearnProj));
R_learn = zeros(length(xp), length(theta_learn));

for i = 1:numLearnProj
    theta_target = theta_learn(i);
    [epsilon, idx] = min(abs(theta_train - theta_target));
    
    if epsilon == 0
        R_learn(:, i) = R_train(:, idx);
    else
        % Vectorized weight calculation for maximum performance
        distances = abs(theta_target - theta_train);
        weights = exp(-(distances / epsilon).^2);
        R_learn(:, i) = (R_train * weights') / sum(weights);
    end
end
time_prop = toc(t_prop);

%% ================= FEEDFORWARD NEURAL NETWORK =================
disp('2/5 Training Feedforward Neural Network...');
t_fnn = tic;
net_fnn = feedforwardnet(10, 'traingd');
net_fnn.trainParam.showWindow = false; % Hides the training GUI
net_fnn = train(net_fnn, theta_train, R_train);

R_fnn = net_fnn(theta_learn);
time_fnn = toc(t_fnn);

%% ================= CUBIC SPLINE INTERPOLATION =================
disp('3/5 Performing Cubic Spline Interpolation...');
t_spline = tic;
R_spline = spline(theta_train, R_train, theta_learn);
time_spline = toc(t_spline);

%% ================= GAUSSIAN PROCESS REGRESSION =================
disp('4/5 Training GPR models (this may take some time)...');
t_gpr = tic;

theta_train_col = theta_train';
theta_learn_col = theta_learn';
R_gpr = zeros(length(xp), length(theta_learn));

for k = 1:length(xp)
    Y = R_train(k, :)';
    gprModel = fitrgp(theta_train_col, Y, ...
        'Basis', 'linear', ...
        'FitMethod', 'exact', ...
        'PredictMethod', 'exact');
    
    predictions = predict(gprModel, theta_learn_col);
    R_gpr(k, :) = predictions';
end
time_gpr = toc(t_gpr);

%% ================= RADIAL BASIS FUNCTION NETWORK =================
disp('5/5 Training Radial Basis Function Network...');
t_rbf = tic;
net_rbf = newrb(theta_train, R_train, 0, 1, 10, 1); 
R_rbf = sim(net_rbf, theta_learn);
time_rbf = toc(t_rbf);

%% ================= IMAGE RECONSTRUCTION =================
disp('Reconstructing images using inverse Radon transform...');
rec_train  = iradon(R_train, theta_train, 'linear', 'Ram-Lak', 1, sz);
rec_learn  = iradon(R_learn, theta_learn, 'linear', 'Ram-Lak', 1, sz);
rec_fnn    = iradon(R_fnn, theta_learn, 'linear', 'Ram-Lak', 1, sz);
rec_spline = iradon(R_spline, theta_learn, 'linear', 'Ram-Lak', 1, sz);
rec_gpr    = iradon(R_gpr, theta_learn, 'linear', 'Ram-Lak', 1, sz);
rec_rbf    = iradon(R_rbf, theta_learn, 'linear', 'Ram-Lak', 1, sz);

%% ================= ERROR ANALYSIS =================
disp('Calculating Frobenius norms...');
theta_ref = 0:1:179;
[R_ref, ~] = radon(P, theta_ref);
img_ref = iradon(R_ref, theta_ref, 'linear', 'Ram-Lak', 1, sz);

err_train  = norm(rec_train  - img_ref, 'fro');
err_learn  = norm(rec_learn  - img_ref, 'fro');
err_fnn    = norm(rec_fnn    - img_ref, 'fro');
err_spline = norm(rec_spline - img_ref, 'fro');
err_gpr    = norm(rec_gpr    - img_ref, 'fro');
err_rbf    = norm(rec_rbf    - img_ref, 'fro');

%% ================= VISUALIZATION =================
disp('Plotting results...');
close all

% 1. Sinograms Figure
figure('Name', 'Sinogram Comparisons', 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.7]);
annotation('textbox', [0 0.94 1 0.06], 'String', 'Sinogram Reconstructions', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'FontSize', 14, 'FontWeight', 'bold');

subplot(2,3,1); imagesc(theta_train, xp, R_train); colormap(hot); colorbar;
title('Noisy Sinogram'); xlabel('\theta'); ylabel('x''');

subplot(2,3,2); imagesc(theta_learn, xp, R_learn); colormap(hot); colorbar;
title('Proposed Methodology'); xlabel('\theta'); ylabel('x''');

subplot(2,3,3); imagesc(theta_learn, xp, R_spline); colormap(hot); colorbar;
title('Cubic Spline'); xlabel('\theta'); ylabel('x''');

subplot(2,3,4); imagesc(theta_learn, xp, R_gpr); colormap(hot); colorbar;
title('GPR'); xlabel('\theta'); ylabel('x''');

subplot(2,3,5); imagesc(theta_learn, xp, R_fnn); colormap(hot); colorbar;
title('FNN'); xlabel('\theta'); ylabel('x''');

subplot(2,3,6); imagesc(theta_learn, xp, R_rbf); colormap(hot); colorbar;
title('RBF Network'); xlabel('\theta'); ylabel('x''');

% 2. Reconstructions Figure
fig = figure('Name', 'Image Reconstructions', 'Units', 'normalized', ...
    'Position', [0.05, 0.05, 0.9, 0.8]);

% Main Title
annotation('textbox', [0 0.95 1 0.05], 'String', 'RECONSTRUCTED IMAGES (Abdomen)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'FontSize', 16, 'FontWeight', 'bold');

% 2x3 subplot layout for 6 images (no empty slots)
subplot(2,3,1); imshow(rec_train, []); title('Filtered Back Projection');
subplot(2,3,2); imshow(rec_learn, []); title('Proposed Methodology');
subplot(2,3,3); imshow(rec_spline, []); title('Cubic Spline');
subplot(2,3,4); imshow(rec_gpr, []); title('GPR');
subplot(2,3,5); imshow(rec_fnn, []); title('FNN');
subplot(2,3,6); imshow(rec_rbf, []); title('RBF Network');

% Adjust subplot spacing to minimize gaps
set(fig, 'Color', 'white');

%% ================= OUTPUT & CREDITS =================
clc
fprintf('\n======================================================================\n');
fprintf('#### Project supported by Centro de Modelamiento Matematico (Chile) ####\n');
fprintf('======================================================================\n');
disp('RESULTS SUMMARY:');
fprintf('======================================================================\n');

disp('Reconstruction Error (Frobenius Norm):');
fprintf('- Filtered Back Projection (Train): %f\n', err_train);
fprintf('- Proposed Methodology:             %f\n', err_learn);
fprintf('- Feedforward Neural Network (FNN): %f\n', err_fnn);
fprintf('- Cubic Spline Interpolation:       %f\n', err_spline);
fprintf('- Gaussian Process Regression:      %f\n', err_gpr);
fprintf('- Radial Basis Function (RBF):      %f\n', err_rbf);

fprintf('\nComputational Time (seconds):\n');
fprintf('- Proposed Methodology:             %f\n', time_prop);
fprintf('- Feedforward Neural Network (FNN): %f\n', time_fnn);
fprintf('- Cubic Spline Interpolation:       %f\n', time_spline);
fprintf('- Gaussian Process Regression:      %f\n', time_gpr);
fprintf('- Radial Basis Function (RBF):      %f\n', time_rbf);
fprintf('======================================================================\n');