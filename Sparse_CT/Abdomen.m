clc
clear all
close all
disp('####Learning through Diffusion####');
disp('####Sparse Tomographic Reconstruction of the Abdomen####');


trainNumber = input('Please enter the number of training projections: ');
lernNumber = input('Please enter the number of tomographic projections to be learned: ');


filename_char = 'Abdomen.png';
P = imread(filename_char);
P=im2double(P)+0.2*rand(256);
theta =180*rand(1,trainNumber);
theta=sort(theta);
[R,xp] = radon(P,theta);


%%%%%%%%%%%Learning through Diffusion%%%%%%%%%%%
theta_new=180*rand(1,lernNumber);
theta_new=sort(theta_new);
R_new=zeros(length(xp),length(theta_new));


for i=1:lernNumber
    sum_var=zeros(length(xp),1);
    nor_var=0;
    xtemp=theta_new(i);
   [m,I]=min(abs(theta-xtemp));
    Epsilon=-m/log(0.1);
    if Epsilon==0
        R_new(:,i)=R(:,I);    
    else
    for inew=1:length(theta)
        ytemp=theta(inew);
        dist_var=abs(xtemp-ytemp);
        dist_var=power(dist_var/Epsilon,2);
        
        sum_var=sum_var+exp(-dist_var)*R(:,inew);
        nor_var=nor_var+exp(-dist_var);
    end
    R_new(:,i)=sum_var/nor_var;   
    end
end

%%%%%%%%%%%Feedforward neural network%%%%%%%%%%%

net_1 = feedforwardnet(10,'traingd');
net_1 = train(net_1,theta,R);

R_fnn=net_1(theta_new);

%%%%%%%%%%%Cubic spline interpolation%%%%%%%%%%%

R_inter = spline(theta,R,theta_new);



%%%%%%%%%%% End %%%%%%%%%%%

figure
imagesc(theta,xp,R) 
colormap(hot); colorbar
xlabel('\theta'); ylabel('x\prime')
title("Sinogram from training projections")

figure
imagesc(theta_new,xp,R_new) 
colormap(hot); colorbar
xlabel('\theta'); ylabel('x\prime')
title("Sinogram from learned projections")

figure
imagesc(theta_new,xp,R_fnn) 
colormap(hot); colorbar
xlabel('\theta'); ylabel('x\prime')
title("Sinogram from the feedforward neural network")

figure
imagesc(theta_new,xp,R_inter) 
colormap(hot); colorbar
xlabel('\theta'); ylabel('x\prime')
title("Sinogram from the cubic spline interpolation")


Reconstruction_train=iradon(R,theta);
Reconstruction_learned=iradon(R_new,theta_new);
Reconstruction_fnn=iradon(R_fnn,theta_new);
Reconstruction_inter=iradon(R_inter,theta_new);

figure
imshow(P,[])
title("Abdomen")

figure
imshow(Reconstruction_train,[])
title("Reconstructed image from training projections")


figure
imshow(Reconstruction_learned,[])
title("Reconstructed image from learned projections")

figure
imshow(Reconstruction_fnn,[])
title("Feedforward neural network")


figure
imshow(Reconstruction_inter,[])
title("Cubic spline interpolation")



[R_original,xp] = radon(P,1:0.25:180);
R_original=iradon(R_original,1:0.25:180);
 

 v_temp1=norm(Reconstruction_train-R_original,"fro");
 v_temp2=norm(Reconstruction_learned-R_original,"fro");
 v_temp3=norm(Reconstruction_fnn-R_original,"fro");
 v_temp4=norm(Reconstruction_inter-R_original,"fro");
 
 
 
output_display = ['The MSE on the training dataset is ',num2str(v_temp1),', the MSE obtained using the learning approach is ',num2str(v_temp2),', the MSE obtained using the feedforward neural network is ' ,num2str(v_temp3), ', the MSE obtained using the cubic spline interpolation is ' ,num2str(v_temp4),];
 
disp(output_display)