%% Constant Predictor.
% must load('YearPredictionMSD.txt') before any code here.
tic

%% Split data according to developer's train/test sizes.
MSD = YearPredictionMSD;
Ntrain = 463715;
Ntest = 51630;
Ntot = size(MSD,1);
Ytrain = MSD(1:Ntrain,1);
Ytest = MSD(Ntot-Ntest+1:Ntot,1);

%% Predictor.
fhat_Const = mean(Ytrain);

%% Results.
Const_error = sum(abs(Ytest-fhat_Const))/size(Ytest,1);
max_dist = max(abs(Ytest-fhat_Const));
min_dist = min(abs(Ytest-fhat_Const));
std_dist = std(abs(Ytest-fhat_Const));

disp(Const_error)
disp(std_dist)
disp(max_dist)
disp(min_dist)
toc