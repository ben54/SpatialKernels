function [ k ] = spatialKernel(test, train, gamma)
% takes the i,j values of two pixels and returns k(pix1,pix2)
    k = exp(-gamma .* pdist2(test, train, 'euclidean') .^ 2);
end