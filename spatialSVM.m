function [ model ] = spatialSVM(data, labels, kernel, cost)
%spatialSVM Builds an SVM classifier on a sample
%   data is the sample
%   labels are the labels for the sample (can be multiclass)
%   kernel is a kernel FUNCTION (passed using @)
%   cost is the cost for C-SVC
%   Output: Cell containing a one-vs-rest model for each label

    uniqueLabels = unique(labels);
    numLabels = size(uniqueLabels,1);
    N = size(data,1);
    K = zeros(N);
    flags = strcat({'-s 0 -t 4 -b 1 -h 0 -c'},{' '}, {int2str(cost)});
    for i=1:N
        for j=1:N
            K(i,j) = kernel(data(i,:),data(j,:));
        end
    end
    
    K = [(1:N)' K];
    
    model = cell(numLabels,1);
    for k=1:numLabels
        model{k} = svmtrain(double(labels==uniqueLabels(k)), K, flags{1});
%         model{k} = svmtrain(double(labels==uniqueLabels(k)), K, '-t 4 -b 1 -c 100 -h 0');
    end
    
%     %# get probability estimates of test instances using each model
%     prob = zeros(N,numLabels);
%     for k=1:numLabels
%         [~,~,p] = svmpredict(double(testLabels==k), KK_, model{k}, '-b 1');
%         %[~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
%         prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
%     end
end

% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%   4 -- pre-computed kernel (put index in first column of K) -- see http://stackoverflow.com/questions/7715138/using-precomputed-kernels-with-libsvm
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)

