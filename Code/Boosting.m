%% Boosting
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

%% Predictor
t = templateTree;
Ensemble = fitensemble(Xtrain,Ytrain,'LSBoost',100,t);
fhat_Boost = predict(Ensemble,Xtest);

%% Results
Boost_error = sum(abs(Ytest-fhat_Boost))/size(Ytest,1);
max_dist = max(abs(Ytest-fhat_Boost));
min_dist = min(abs(Ytest-fhat_Boost));
std_dist = std(abs(Ytest-fhat_Boost));

disp(Boost_error)
disp(std_dist)
disp(max_dist)
disp(min_dist)
toc
