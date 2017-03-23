function [ pred, model ] = spatialSVMPredict(kernel, trainData, labels, testData, cost, gamma)
%spatialSVMPredict Uses models built in spatialSVM to predict test data
%   models is result from spatialSVM
%   kernel is a FUNCTION
%   trainData is MxD data - used to build KK
%   testData is NxD, where N is number of test instances and D is dim
%   Output: Nx1 array of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOT OPTIMIZED TO ONLY USE SUPPORT VECTORS OF MODELS... YET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [M, ~] = size(testData);
    [N, ~] = size(trainData);
    
    flags = strcat({'-s 0 -t 4 -b 1 -h 0 -q 1 -c'}, {' '}, ...
                    {num2str(cost, '%f')});
    
    % build kernel matrix and re-train the model with best params
    K = kernel(trainData, trainData, gamma);
    K = [(1:N)' K];
        
    model = svmtrain(double(labels), K, flags{1});
% labels                    
    % build the test kernel matrix and predict using trained model
    KK = kernel(testData, trainData, gamma);
    KK = [(1:M)', KK];
    
    testLabels = double(zeros(M, 1));
    [ypred, ~, ~] = svmpredict(testLabels, KK, model, '-b 1 -q 1');
    
    % libsvm does not respect input label order, so swap according to
    % order in model.Label
    pred = ypred;
%     for i=1:size(model.Label, 1)
%         mask = ypred == i;
%         pred(mask) = model.Label(i);
%     end
end

