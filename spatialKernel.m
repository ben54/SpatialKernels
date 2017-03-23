function [ k ] = spatialKernel(pix1, pix2)
%spatialKernel Takes the i,j values of two pixels and returns k(pix1,pix2)
%     sigma = .1;
%     k = exp(norm(pix1-pix2)^2/(2*sigma^2));
    k = dot(pix1, pix2);
end

