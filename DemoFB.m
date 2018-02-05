%%% Wrapper for Foreground-Background task in video layering

%%%If you use this code please cite the following paper 
%%%[1] A Fast and Memory-Efficient Algorithm for Robust PCA (MEROP), 
%%%     Praneeth Narayanamurthy and Namrata Vaswani, ICASSP 2018.


%%Read video
clear;
clc;
close all

addpath('YALL1_v1.4/');
addpath('PROPACK/');

load('Data/Curtain.mat');
%I = M;


%% Training data processing
%%option 1 -- init using batch RPCA
t_train = 400;
TrainData = I(:, 1 : t_train);
rank_init = 50;

L_hat_init = ncrpca(TrainData, rank_init);

mu = mean(L_hat_init, 2);


[Utemp, Stemp, ~] = svd(1 / sqrt(t_train) * (L_hat_init - ...
    repmat(mu, 1, t_train)));
ss1 = diag(Stemp);
L_init = Utemp(:, 1 : rank_init);

%%option 2 -- init using outlier free data
% mu = mean(DataTrain, 2);
% t_train = size(DataTrain, 2);
% [Utemp, Stemp, ~] = svd(1 / sqrt(t_train) * ...
%     (DataTrain - repmat(mu, 1, t_train)));
% ss1 = diag(Stemp);
% b = 0.95;
% rank_init = min(find(cumsum(ss1.^2) >= b * sum(ss1.^2)));
% L_init = Utemp(:, 1 : rank_init);

fprintf('Initialized\n');


%% Call to MERoP

K = 3;
alpha = 60;

tic
fprintf('alpha = %d\tK = %d\n', alpha, K);
[BG, FG, L_hat, S_hat, T_hat, t_hat, P_track_full, P_track_new] ...
    = MERoP(I(:, t_train + 1 : end), L_init, mu, alpha, K);
toc



VidName = 'DemoOutput';
DisplayVideo(I(:, t_train + 1 : end), FG, BG, T_hat, imSize, VidName);