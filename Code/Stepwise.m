%% Forward-Backward Stepwise Subset Selection.
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
[b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(Xtrain,Ytrain,...
    'display','off');
Xte = Xtest(:,inmodel);
fhat_FBS = Xte*b(inmodel)+stats.intercept;

%% Results.
FBS_error = sum(abs(Ytest-fhat_FBS))/size(Ytest,1);
max_dist = max(abs(Ytest-fhat_FBS));
min_dist = min(abs(Ytest-fhat_FBS));
std_dist = std(abs(Ytest-fhat_FBS));

disp(FBS_error)
disp(std_dist)
disp(max_dist)
disp(min_dist)
toc