%% Lasso
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
lambda = 6.9171e-04;
[beta_Lasso, FitInfo] = lasso([ones(size(Xtrain,1),1),Xtrain],Ytrain,'Lambda',lambda);
fhat_Lasso = [ones(size(Xtest,1),1),Xtest]*beta_Lasso+FitInfo.Intercept;

%% Results.
Lasso_error = sum(abs(Ytest-fhat_Lasso))/size(Ytest,1);
max_dist = max(abs(Ytest-fhat_Lasso));
min_dist = min(abs(Ytest-fhat_Lasso));
std_dist = std(abs(Ytest-fhat_Lasso));

disp(Lasso_error)
disp(std_dist)
disp(max_dist)
disp(min_dist)
toc

%% Cross validation to determine optimal lambda.
% Keep commented out unless optimal lambda computation is desired. Cross 
% validation here takes roughly 3 hours on a quad core, 2 GHz Intel Core i7
% using the 'UseParallel' option.

% train on 4/5th and test on 1/5th.
% K = 5;
% CVO = cvpartition(Ntrain,'kfold',K);
% opt = statset('UseParallel',true);

% [beta_Lasso, FitInfo] = lasso([ones(size(Xtrain,1),1),Xtrain],Ytrain,...
%     'CV',CVO,'Options',opt);%,'Lambda',lambda);
% fhat_Lasso = zeros(Ntest,length(FitInfo.Lambda));
% for j=1:length(FitInfo.Lambda)
% fhat_Lasso(:,j) = [ones(size(Xtest,1),1),Xtest]*beta_Lasso(:,j)+FitInfo.Intercept(j);
% end
 
% plot(1:length(FitInfo.Lambda),FitInfo.MSE,'r*')
% disp(FitInfo.LambdaMinMSE);
% disp(FitInfo.IndexMinMSE);

