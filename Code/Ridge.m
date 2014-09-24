%% Ridge regression.
% must load('YearPredictionMSD.txt') before any code here.
tic

%% Split data according to developer's train/test sizes.
MSD = YearPredictionMSD;
Ntrain = 463715;                                % number of training data
Ntest = 51630;                                  % number of test data
Ntot = size(MSD,1);                             % total number of data
p = size(MSD,2)-1;                              % size of feature space
Xtrain = MSD(1:Ntrain,2:size(MSD,2));           % training input
Ytrain = MSD(1:Ntrain,1);                       % training output
Xtest = MSD(Ntot-Ntest+1:Ntot,2:size(MSD,2));   % test input
Ytest = MSD(Ntot-Ntest+1:Ntot,1);               % test output

%% Random subsample of training and testing sets.
% keep commented out to run on entire data set.

% rNtrain = 10000;
% rIdx_train = randsample(Ntrain,rNtrain);
% rNtest = 500;
% rIdx_test = randsample(Ntest,rNtest);
% 
% Xtrain = Xtrain(rIdx_train,:);
% Ytrain = Ytrain(rIdx_train);
% Xtest = Xtest(rIdx_test,:);
% Ytest = Ytest(rIdx_test);
% 
% Ntrain = size(Xtrain,1);
% Ntest = size(Xtest,1);
% Ntot = Ntrain + Ntest;

%% Predictor.
% loop through different lambda values to determine optimal lambda.
% lamb = 0:0.00005:0.001;
% for i = 1:length(lamb)
% lambda = lamb(i);

lambda = 0.0001;
beta_Ridge = ridge(Ytrain,Xtrain,lambda,0);
fhat_Ridge = [ones(size(Xtest,1),1),Xtest]*beta_Ridge;

%% Results.
Ridge_error = sum(abs(Ytest-fhat_Ridge))/size(Ytest,1);
max_dist = max(abs(Ytest-fhat_Ridge));
min_dist = min(abs(Ytest-fhat_Ridge));
std_dist = std(abs(Ytest-fhat_Ridge));

disp(Ridge_error)
disp(std_dist)
disp(max_dist)
disp(min_dist)
toc
%end
