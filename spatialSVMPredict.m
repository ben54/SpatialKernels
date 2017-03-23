function [ pred ] = spatialSVMPredict(models, kernel, trainData, testData)
%spatialSVMPredict Uses models built in spatialSVM to predict test data
%   models is result from spatialSVM
%   kernel is a FUNCTION
%   trainData is MxD data - used to build KK
%   testData is NxD, where N is number of test instances and D is dim
%   Output: Nx1 array of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOT OPTIMIZED TO ONLY USE SUPPORT VECTORS OF MODELS... YET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [N,D] = size(testData);
    [M,~] = size(trainData);
    numLabels = size(models,1);
    KK = zeros(N,M);
    for i=1:N
        for j=1:M
            KK(i,j) = kernel(testData(i,:),trainData(j,:));
        end
    end
    KK = [(1:N)' KK];
     %# get probability estimates of test instances using each model
    prob = zeros(N,numLabels);
    testLabels = zeros(N,1);
    for k=1:numLabels
        [~,~,p] = svmpredict(testLabels, KK, models{k}, '-b 1 -q 1');
%         [~,~,p] = svmpredict(double(testLabels==k), KK, models{k}, '-b 1');
        prob(:,k) = p(:,models{k}.Label==1);    %# probability of class==k
    end

    %# predict the class with the highest probability
    [~,pred] = max(prob,[],2);
  
end

