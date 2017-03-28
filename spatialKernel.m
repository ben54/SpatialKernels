function [ k ] = spatialKernel(test, train, gamma)
%   Takes two sets of points (test and train) and gamma and returns 
%   k(x_i, x_j) for each x_i in test and x_j in train
%   the similarity function used is the RBF or Gaussian 
    k = exp(-gamma .* pdist2(test, train, 'euclidean') .^ 2);
end