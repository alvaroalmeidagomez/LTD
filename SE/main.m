clearvars
close all
clc

disp(' ')
disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp(' Interpolating over the Logarithmic Cube [0,1] x[0,1]')
disp('==============================================')
disp(' ')

m      = input('Enter the number of input points on the Cube: ');
m_new  = input('Enter the number of interpolation points on the cube: ');

% Create dataset
D = rand(m, 2);
f = sin(10*pi*D(:,1)) .* cos(10*pi*D(:,2));

%%%%%%%%%%%%%%%%%% Testing data-set %%%%%%%%%%%%%%%%%%
sol = zeros(m_new, 1);
D_new = rand(m_new, 2);

Tempvar1 = tic;

% VECTORIZED APPROACH: pdist2 calculates distances much faster than loops
dist_matrix = pdist2(D_new, D); 
[min_dist, I] = min(dist_matrix, [], 2);

for i = 1:m_new
    epsilon = min_dist(i);
    
    if epsilon == 0
        sol(i) = f(I(i));
    else
        % dist_matrix(i,:) contains the distances from D_new(i) to all D
        dist_var = (dist_matrix(i, :) / epsilon).^2;
        weights = exp(-dist_var)'; 
        sol(i) = sum(weights .* f) / sum(weights);
    end
end
Tempvar1 = toc(Tempvar1);

%%%%%%%%%%% Feedforward neural network %%%%%%%%%%%
Tempvar2b = tic;

net_1 = feedforwardnet(10, 'traingd');
% Optional: Hide the training window to avoid pop-ups during execution
net_1.trainParam.showWindow = false; 
net_1 = train(net_1, D', f');

Tempvar2tra = toc(Tempvar2b);

f_fnn = net_1(D_new');

Tempvar2 = toc(Tempvar2b);
%%%%%%%%%%% End %%%%%%%%%%%

%%%%%%%%%%% Gaussian process regression %%%%%%%%%%%
Tempvar3 = tic;

gprMdl1 = fitrgp(D, f, 'Basis', 'linear', ...
      'FitMethod', 'exact', 'PredictMethod', 'exact');
  
Tempvar3tra = toc(Tempvar3);

[f_GP, Y_sd, Y_int] = predict(gprMdl1, D_new);

Tempvar3 = toc(Tempvar3);
%%%%%%%%%%% End %%%%%%%%%%%

%%%%%%%%%%% Radial basis function network %%%%%%%%%%%
Tempvar4 = tic;

netRB = newrb(D', f');

Tempvar4tra = toc(Tempvar4);

f_RB = sim(netRB, D_new');

Tempvar4 = toc(Tempvar4);
%%%%%%%%%%% End %%%%%%%%%%%


%%%%%%%%%%% PLOTTING: ALL IN ONE FIGURE %%%%%%%%%%%
% Create a single large figure window
figure('Name', 'Interpolation Results', 'Units', 'Normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

% 1. Function f(x)
subplot(2, 3, 1)
scatter(D(:,1), D(:,2), 40, f, 'filled')
colorbar;
title('Function f(x)')
xlabel('First coordinate'); ylabel('Second coordinate')
xlim([0 1]); ylim([0 1])

% 2. Proposed Methodology
subplot(2, 3, 2)
scatter(D(:,1), D(:,2), 40, f, 'filled')
hold on
scatter(D_new(:,1), D_new(:,2), 15, sol, 'filled')
title('Proposed methodology')
xlabel('First coordinate'); ylabel('Second coordinate')
colorbar;
xlim([0 1]); ylim([0 1])

% 3. Feedforward Neural Network
subplot(2, 3, 3)
scatter(D(:,1), D(:,2), 40, f, 'filled')
hold on
scatter(D_new(:,1), D_new(:,2), 15, f_fnn, 'filled')
title('Feedforward neural network')
xlabel('First coordinate'); ylabel('Second coordinate')
colorbar;
xlim([0 1]); ylim([0 1])

% 4. Gaussian Process Regression
subplot(2, 3, 4)
scatter(D(:,1), D(:,2), 40, f, 'filled')
hold on
scatter(D_new(:,1), D_new(:,2), 15, f_GP, 'filled')
title('Gaussian process regression')
xlabel('First coordinate'); ylabel('Second coordinate')
colorbar;
xlim([0 1]); ylim([0 1])

% 5. Radial Basis Function Network
subplot(2, 3, 5)
scatter(D(:,1), D(:,2), 40, f, 'filled')
hold on
scatter(D_new(:,1), D_new(:,2), 15, f_RB, 'filled')
title('Radial basis function network')
xlabel('First coordinate'); ylabel('Second coordinate')
colorbar;
xlim([0 1]); ylim([0 1])


%%%%%%%%%%% ERROR CALCULATION & DISPLAY %%%%%%%%%%%
fx = sin(10*pi*D_new(:,1)) .* cos(10*pi*D_new(:,2));

% Transposed variables mathematically aligned to prevent dimension mismatch errors
v_temp1 = norm(fx - sol);
v_temp2 = norm(fx - f_fnn');
v_temp3 = norm(fx - f_GP);
v_temp4 = norm(fx - f_RB');

clc

disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp('==============================================')

disp('==============================================')
disp('Computational results on the Cube [0,1] x[0,1]')
disp('==============================================')

output_display = sprintf(['Squared Error Results:\n', ...
    '  - Proposed Methodology:         %.4f\n', ...
    '  - Feedforward Neural Network:   %.4f\n', ...
    '  - Gaussian Process Regression:  %.4f\n', ...
    '  - Radial Basis Function:        %.4f'], ...
    v_temp1, v_temp2, v_temp3, v_temp4);

output_display_time = sprintf(['Computational time (seconds):\n', ...
    '  - Proposed Methodology:         %.4f\n', ...
    '  - Feedforward Neural Network:   %.4f\n', ...
    '  - Gaussian Process Regression:  %.4f\n', ...
    '  - Radial Basis Function:        %.4f'], ...
 Tempvar1, Tempvar2, Tempvar3, Tempvar4);

disp(output_display)
disp('==============================================')
disp('==============================================')

disp(output_display_time)

disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp('==============================================')