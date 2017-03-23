clear
close all
% rand('state', 0);
rng(0);

filePath = '/Users/ben/Dropbox/Projects/MKLPixelClassification';
cd(filePath)
img = imread('Data/etm/2016-03-20-AllBands-Clipped.tif');
%m1 = csvread(strcat(filePath,'tn22.csv'));
% imshow(img(:,:,6:8))

testImage = zeros(500);
% imshow(testImage)

numTrain = 500;
xTrainData = randsample(500,500,true);
yTrainData = randsample(500,500,true);
classTrainData = randsample(6,500,true);

