function [ best_cost, best_gamma ] = spatialSVM(data, labels, kernel)
%spatialSVM Builds an SVM classifier on a sample
%   data is the sample
%   labels are the labels for the sample (can be multiclass)
%   kernel is a kernel FUNCTION (passed using @)
%   cost is the cost for C-SVC
%   Output: Cell containing a one-vs-rest model for each label

    N = size(data, 1);
    % random search over hyperparameter space
    a = -5; b = 11;
    r = a + (b - a) .* rand(10, 1);
    costs = 10 .^ r;
    a = -9; b = 3;
    r = a + (b - a) .* rand(10, 1);
    gammas = 10 .^ r;
    best_val_acc = 0;
    
    for gamma=gammas'
        Kfull = zeros(N, N);
        Kfull = kernel(data, data, gamma);
        for cost=costs'
            flags = strcat({'-t 4 -b 0 -h 0 -q 1 -c'}, {' '}, ...
                {num2str(cost, '%f')});
            breaks = [1:floor(N / 10):N N+1];
            accs = zeros(1, 10);
            for fold=1:10
                % partition data into training and validation sets
                valIdx = breaks(fold):(breaks(fold + 1) - 1);
                trainIdx = setdiff(1:N, valIdx);
                y_val = labels(valIdx);

                K = Kfull(trainIdx, trainIdx);
                K = [(1:size(trainIdx, 2))' K];
                KK = Kfull(valIdx, trainIdx);
                KK = [(1:size(valIdx, 2))' KK];

                model = svmtrain(double(labels(trainIdx)), K, flags{1});
                [~, acc, ~] = svmpredict(double(y_val), KK, model, '-q 1');
                accs(fold) = acc(1) / 100.0;
            end
            val_acc = mean(accs);
            if(val_acc > best_val_acc)
                best_val_acc = val_acc;
                best_gamma = gamma;
                best_cost = cost;
            end
        end
    end
    fprintf('Best gamma is %f\n', best_gamma);
    fprintf('Best cost accuracy is %f\n', best_cost);
    fprintf('Best validation accuracy is %f\n', best_val_acc);
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

