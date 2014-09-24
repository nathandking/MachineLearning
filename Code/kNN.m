%% k-Nearest-Neighbours
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
K=50;
NS = ExhaustiveSearcher(Xtrain);
Idx = knnsearch(NS,Xtest,'k',K);

fhat_kNN = zeros(size(Ytest,1),1);
for i=1:size(Ytest,1)
fhat_kNN(i)=(1/K)*sum(Ytrain(Idx(i,:)));
end

%% Results.
kNN_error = sum(abs(Ytest-fhat_kNN))/size(Ytest,1);
max_dist = max(abs(Ytest-fhat_kNN));
min_dist = min(abs(Ytest-fhat_kNN));
std_dist = std(abs(Ytest-fhat_kNN));

disp(kNN_error)
disp(std_dist)
disp(max_dist)
disp(min_dist)
toc