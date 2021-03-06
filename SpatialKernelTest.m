clear all; close all; clc;

library = 'libsvm'; % to use libsvm/matlab library

% Generate testing surface. Then, test data is test surface evaulated at 
% fine interval
x = linspace(-5, 5, 100);
y = linspace(-5, 5, 100);
[X, Y] = meshgrid(x, y);
X_flat = reshape(X, [100 * 100, 1]);
Y_flat = reshape(Y, [100 * 100, 1]);
X_test = [X_flat Y_flat];

% Generate training data
N = 300;
% numlabels = 3;
each = floor(N/3);
X_1 = mvnrnd([0.3 0.3], [0.1 0; 0 0.1], each);
X_2 = mvnrnd([-0.5 0.7], [0.23 0.05; 0.05 0.45], each);
X_3 = mvnrnd([0 -0.7], [0.6 -0.005; -0.005 0.1], each);
X_train = [X_1; X_2; X_3];
% X_train = rand(N, 2) * 10 - ones(N, 2) * 5;

y_train = [repmat(3, each, 1); repmat(1, each, 1); repmat(2, each, 1)];
% y_train = randi(numlabels, N, 1);
 
if(strcmp(library, 'matlab') == 1)
    t = templateSVM('KernelFunction', 'gaussian');
    % build model
    SVMModel = fitcecoc(X_train, y_train, 'Learners', t);
    % predict on test
    y_pred = predict(SVMModel, X_test);
elseif(strcmp(library, 'libsvm') == 1)
    % optimize hyperparameters over k-fold CV
    [best_cost, best_gamma] = optimizeParams(X_train, y_train, ...
        @spatialKernel);
    % rebuild model with best params and predict on test
    [y_pred, model] = spatialSVMPredict(@spatialKernel, X_train, y_train, ...
        X_test, best_cost, best_gamma);
end

% plot the domain surface
Z = reshape(y_pred, [100, 100]);
figure
subplot(2, 1, 1)
contour(X, Y, Z)
hold on
% overlay the training data
gscatter(X_train(:, 1), X_train(:, 2), y_train, 'brk', 'xos')
hold on
subplot(2, 1, 2)
surf(X, Y, Z)
