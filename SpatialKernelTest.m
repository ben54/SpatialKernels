clear all

x = linspace(-5,5);
y = linspace(-5,5);
[X,Y] = meshgrid(x,y);

X_flat = reshape(X,[100*100,1]);
Y_flat = reshape(Y,[100*100,1]);
test = [X_flat Y_flat];

numSamples = 20;
sample = rand(numSamples,2)* 10 - 5*ones(numSamples,2);
sampleLabels = randi(3,numSamples,1);
% sampleLabels = [1 2 3];
% models = spatialSVM(sample, sampleLabels, @spatialKernel, 1);
% SVMModel = fitcsvm(sample,sampleLabels,'KernelFunction','Linear');

t = templateSVM('KernelFunction','gaussian')
SVMModel = fitcecoc(sample,sampleLabels,'Learners',t);

% labels = spatialSVMPredict(models, @spatialKernel, sample, test);
labels = predict(SVMModel,test);
Z = reshape(labels,[100,100]);

figure
subplot(2,1,1)
contour(X,Y,Z)
hold on
gscatter(sample(:,1),sample(:,2),sampleLabels,'brk','xos')
% hold on
subplot(2,1,2)
gscatter(sample(:,1),sample(:,2),sampleLabels,'brk','xos')
surf(X,Y,Z)

% a = reshape(reshape(X,[100*100,1]),[100,100]);