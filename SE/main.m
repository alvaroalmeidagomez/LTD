clear all
close all
clc
disp('####Learning through Diffusion####');
disp('####Learning Over the Logarithmic Spiral####');


m     = input('Enter the number of training points on the logarithmic spiral: ');
m_new = input('Enter the number of generated points to be trained on the cube: ');
m_new2= input('Enter the number of generated points to be trained on the spiral: ');


% Create dataset
t =  rand(m, 1);
D=exp(t).*[ cos(2*pi*t) sin(2*pi*t)];
f=t;

%%%%%%%%%%%%%%%%%%Testing data-set%%%%%%%%%%%%%%%%%%
sol=zeros(m_new,1);
D_new=exp(1)*(2*rand(m_new,2)-1);


for i=1:m_new
    sum_var=0;
    nor_var=0;
    xtemp=D_new(i,:);
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
     sol(i)=sum_var/nor_var;
     
   end
end


%%%%%%%%%%%Feedforward neural network%%%%%%%%%%%

net_1 = feedforwardnet(10,'traingd');
net_1 = train(net_1,transpose(D),transpose(f));

f_fnn=net_1(transpose(D_new));

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
title('Function learned through the proposed methodology')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])


figure
scatter(D(:,1),D(:,2),40,f,'filled')
hold on
scatter(D_new(:,1),D_new(:,2),5,f_fnn,'filled')
title('Function learned through the feedfoward neural network')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])

%%%%%%%%%%Part 2 %%%%%%%%%%

%%%%%%%%%%%%%%%%%%Testing data-set%%%%%%%%%%%%%%%%%%
sol2=zeros(m_new2,1);
D_new2 =  rand(m_new2, 1);
D_new2=exp(D_new2).*[ cos(2*pi*D_new2) sin(2*pi*D_new2)];



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


%%%%%%%%%%%Feedforward neural network%%%%%%%%%%%

net_1 = feedforwardnet(10,'traingd');
net_1 = train(net_1,transpose(D),transpose(f));

f_fnn2=net_1(transpose(D_new2));

%%%%%%%%%%%End %%%%%%%%%%%

figure
scatter(D_new2(:,1),D_new2(:,2),40,sol2,'filled')
title('Function learned through the proposed methodology')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])


figure
scatter(D_new2(:,1),D_new2(:,2),40,f_fnn2,'filled')
title('Function learned through the feedfoward neural network')
xlabel('First coordinate') 
ylabel('Second coordinate') 
c=colorbar;
xlim([-exp(1) exp(1)])
ylim([-exp(1) exp(1)])


fx=log(vecnorm(transpose(D_new2)));
v_temp1=norm(fx-transpose(sol2));
v_temp2=norm(fx-transpose(f_fnn2));


output_display = ['The squared error using the proposed methodology is ',num2str(v_temp1),', the squared error obtained using the feedfoward neural network ',num2str(v_temp2),];
disp(output_display)


