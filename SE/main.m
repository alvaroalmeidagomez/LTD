clear all
close all
clc
disp(' ')
disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp(' Interpolating over the Logarithmic Spiral')
disp('==============================================')
disp(' ')


m      = input('Enter the number of input points on the logarithmic spiral: ');
m_new  = input('Enter the number of interpolation points on the cube: ');
m_new2 = input('Enter the number of interpolation points on the spiral: ');


% Create dataset
t =  rand(m, 1);
D=exp(t).*[ cos(4*pi*t) sin(4*pi*t)];
f=t;

%%%%%%%%%%%%%%%%%%Testing data-set%%%%%%%%%%%%%%%%%%
sol=zeros(m_new,1);
D_new=exp(1)*(2*rand(m_new,2)-1);

Tempvar1=tic;
for i=1:m_new
    sum_var=0;
    nor_var=0;
    xtemp=D_new(i,:);
    var_temp=transpose(D-xtemp);
    var_temp=vecnorm(var_temp);
    [mivar , I]=min(var_temp);
    epsilon=mivar;
   
   if mivar==0
    sol(i)=f(I);
   else
       
    for inew=1:m
        ytemp=D(inew,:);
        dist_var=norm(xtemp-ytemp);
        dist_var=power(dist_var/epsilon,2);
              
        sum_var=sum_var+exp(-dist_var)*f(inew);
        nor_var=nor_var+exp(-dist_var);    
    end
     sol(i)=sum_var/nor_var;
     
   end
end
Tempvar1=toc(Tempvar1);


%%%%%%%%%%%Feedforward neural network%%%%%%%%%%%
Tempvar2b=tic;

net_1 = feedforwardnet(10,'traingd');
net_1 = train(net_1,transpose(D),transpose(f));

Tempvar2tra=toc(Tempvar2b);


f_fnn=net_1(transpose(D_new));

Tempvar2=toc(Tempvar2b);

%%%%%%%%%%%End %%%%%%%%%%%

%%%%%%%%%%%Gaussian process regression%%%%%%%%%%%

Tempvar3=tic;

gprMdl1 = fitrgp(D,f,'Basis','linear',...
      'FitMethod','exact','PredictMethod','exact');
  
Tempvar3tra=toc(Tempvar3);

[f_GP, Y_sd, Y_int] = predict(gprMdl1, D_new);


Tempvar3=toc(Tempvar3);

%%%%%%%%%%%End %%%%%%%%%%%

%%%%%%%%%%% Radial basis function network%%%%%%%%%%%

Tempvar4=tic;


netRB = newrb(transpose(D),transpose(f));

Tempvar4tra=toc(Tempvar4);

f_RB = sim(netRB,transpose(D_new));

Tempvar4=toc(Tempvar4);

%%%%%%%%%%%End %%%%%%%%%%%



figure
scatter(D(:,1),D(:,2),40,f,'filled')
c=colorbar;
title('Function f(x)')
xlabel('First coordinate') 
ylabel('Second coordinate') 
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])


figure
scatter(D(:,1),D(:,2),40,f,'filled')
hold on
scatter(D_new(:,1),D_new(:,2),5,sol,'filled')
title('Interpolation through the proposed methodology')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])


figure
scatter(D(:,1),D(:,2),40,f,'filled')
hold on
scatter(D_new(:,1),D_new(:,2),5,f_fnn,'filled')
title('Interpolation through the feedfoward neural network')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])

figure
scatter(D(:,1),D(:,2),40,f,'filled')
hold on
scatter(D_new(:,1),D_new(:,2),5,f_GP,'filled')
title('Interpolation using Gaussian process regression')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])

figure
scatter(D(:,1),D(:,2),40,f,'filled')
hold on
scatter(D_new(:,1),D_new(:,2),5,f_RB,'filled')
title('Interpolation using a radial basis function network')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%Testing data-set%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sol2=zeros(m_new2,1);
D_new2 =  rand(m_new2, 1);
D_new2=exp(D_new2).*[ cos(4*pi*D_new2) sin(4*pi*D_new2)];


Tempvar11=tic;
for i=1:m_new2
    sum_var=0;
    nor_var=0;
    xtemp=D_new2(i,:);
    var_temp=transpose(D-xtemp);
    var_temp=vecnorm(var_temp);
    [mivar , I]=min(var_temp);
    epsilon=-mivar/log(0.1);
   
   if mivar==0
    sol(i)=f(I);
   else
       
    for inew=1:m
        ytemp=D(inew,:);
        dist_var=norm(xtemp-ytemp);
        dist_var=power(dist_var/epsilon,2);
              
        sum_var=sum_var+exp(-dist_var)*f(inew);
        nor_var=nor_var+exp(-dist_var);
        
    end
     sol2(i)=sum_var/nor_var;
     
   end
end
Tempvar11=toc(Tempvar11);

%%%%%%%%%%%Feedforward neural network%%%%%%%%%%%

Tempvar22=tic;

f_fnn2=net_1(transpose(D_new2));

Tempvar22=toc(Tempvar22)+Tempvar2tra;

%%%%%%%%%%%End %%%%%%%%%%%

%%%%%%%%%%%Gaussian process regression%%%%%%%%%%%
Tempvar33=tic;

[f_GP2, Y_sd2, Y_int2] = predict(gprMdl1, D_new2);

Tempvar33=toc(Tempvar33)+Tempvar3tra;

%%%%%%%%%%%End %%%%%%%%%%%

%%%%%%%%%%% Radial basis function network%%%%%%%%%%%
Tempvar44=tic;

f_RB2 = sim(netRB,transpose(D_new2));

Tempvar44=toc(Tempvar44)+Tempvar4tra;
%%%%%%%%%%%End %%%%%%%%%%%




figure
scatter(D_new2(:,1),D_new2(:,2),40,sol2,'filled')
title('Interpolation through the proposed methodology')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])


figure
scatter(D_new2(:,1),D_new2(:,2),40,f_fnn2,'filled')
title('Interpolation through the feedfoward neural network')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])

figure
scatter(D_new2(:,1),D_new2(:,2),40,f_GP2,'filled')
title('Interpolation Using Gaussian Process Regression')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])

figure
scatter(D_new2(:,1),D_new2(:,2),40,f_RB2,'filled')
title('Interpolation using a radial basis function network')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])




fx=log(vecnorm(transpose(D_new2)));
v_temp1=norm(fx-transpose(sol2));
v_temp2=norm(fx-f_fnn2);
v_temp3=norm(fx-transpose(f_GP2));
v_temp4=norm(fx-f_RB2);

clc

disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp('==============================================')

disp('==============================================')
disp('Computational results on the cube')
disp('==============================================')



output_display_time0 = sprintf(['Computational time (seconds)::\n', ...
    '  - Proposed Methodology:         %.4f\n', ...
    '  - Feedforward Neural Network:   %.4f\n', ...
    '  - Gaussian Process Regression:  %.4f\n', ...
    '  - Radial Basis Function:        %.4f'], ...
    Tempvar1, Tempvar2, Tempvar3, Tempvar4);



disp(output_display_time0)

disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp('==============================================')

disp('==============================================')
disp('Computational results on the curve')
disp('==============================================')




output_display = sprintf(['Squared Error Results:\n', ...
    '  - Proposed Methodology:         %.4f\n', ...
    '  - Feedforward Neural Network:   %.4f\n', ...
    '  - Gaussian Process Regression:  %.4f\n', ...
    '  - Radial Basis Function:        %.4f'], ...
    v_temp1, v_temp2, v_temp3, v_temp4);


output_display_time = sprintf(['Computational time (seconds)::\n', ...
    '  - Proposed Methodology:         %.4f\n', ...
    '  - Feedforward Neural Network:   %.4f\n', ...
    '  - Gaussian Process Regression:  %.4f\n', ...
    '  - Radial Basis Function:        %.4f'], ...
    Tempvar11, Tempvar22, Tempvar33, Tempvar44);



disp(output_display)
disp('==============================================')
disp('==============================================')

disp(output_display_time)

disp('==============================================')
disp(' Project supported by the Centro de Modelamiento Matematico (Chile)')
disp('==============================================')



