function [ best_cost, best_gamma ] = spatialSVM(data, labels, kernel)
%   Performs random search over parameter space to determine optimal values
%   using k-fold cross validation
%   Inputs:
%   data is the NxD training data
%   labels is the Nx1 array of training class labels
%   kernel is a function that computes Kernel matrix betw 2 sets of points
%   cost is the cost for C-SVC
%   Output: Cell containing a one-vs-rest model for each label

    N = size(data, 1);
    % random search over hyperparameter space
    numCosts = 10; numGammas = 10;
    a = -5; b = 11;
    r = a + (b - a) .* rand(numCosts, 1);
    costs = 10 .^ r;
    a = -9; b = 3;
    r = a + (b - a) .* rand(numGammas, 1);
    gammas = 10 .^ r;
    best_val_acc = 0;
    
    % cartesian product between parameter arrays
    [G, C] = meshgrid(gammas, costs);
    params = [G(:) C(:)];
    
    numFolds = 20;
    
    % for each combination of parameters, get k-fold cv accuracy
    for i=1:size(params, 1)
        gamma = params(i, 1);
        if mod(i - 1, numCosts) == 0
            Kfull = zeros(N, N);
            Kfull = kernel(data, data, gamma);
        end
        cost = params(i, 2);
        flags = strcat({'-t 4 -h 0 -c '}, ...
            {num2str(cost, '%f')}, {' -q 1'});
        breaks = round(linspace(1, N + 1, numFolds + 1));
        accs = zeros(1, numFolds);
        for fold=1:numFolds
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
        fprintf('Gamma is %.9f, cost is %.9f, validation accuracy is %f\n', ...
            gamma, cost, val_acc);
        if(val_acc > best_val_acc)
            best_val_acc = val_acc;
            best_gamma = gamma;
            best_cost = cost;
        end
    end
    fprintf('Best gamma is %.9f\n', best_gamma);
    fprintf('Best cost is %.9f\n', best_cost);
    fprintf('Best validation accuracy is %f\n', best_val_acc);
end