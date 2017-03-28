function [ ypred, model ] = spatialSVMPredict(kernel, trainData, ...
    labels, testData, cost, gamma)
%   Uses best parameters from spatialSVM to re-train 
%   the model and use it to predict
%   Inputs:
%   kernel is a function that computes Kernel matrix betw 2 sets of points
%   labels is the Nx1 array of training class labels
%   trainData is NxD training data
%   testData is MxD, where M is number of test instances and D is dim
%   cost is the best cost from spatialSVM
%   gamma is the best gamma from spatialSVM
%   Outputs:
%   ypred is a Mx1 array of predicted test labels
%   model is the libsvm C-SVM model struct built

    [M, ~] = size(testData);
    [N, ~] = size(trainData);
    
    flags = strcat({'-s 0 -t 4 -b 1 -h 0 -c '}, ...
                    {num2str(cost, '%f')}, {' -q 1'});
    
    % build kernel matrix and re-train the model with best params
    K = kernel(trainData, trainData, gamma);
    K = [(1:N)' K];
    
    model = svmtrain(double(labels), K, flags{1});
                 
    % build the test kernel matrix and predict using trained model
    KK = kernel(testData, trainData, gamma);
    KK = [(1:M)', KK];
    
    testLabels = double(zeros(M, 1));
    [ypred, ~, ~] = svmpredict(testLabels, KK, model, '-b 1 -q 1');
end