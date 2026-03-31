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


% Create dataset
D = rand(m, 2);
f = sin(10*pi*D(:,1)) .* cos(10*pi*D(:,2));


scatter(D(:,1), D(:,2), 40, f, 'filled')
colorbar;
title('Function f(x)')
xlabel('First coordinate'); ylabel('Second coordinate')
xlim([0 1]); ylim([0 1])