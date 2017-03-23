function [ output ] = RELUDiff( input )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here
    if input <= 0
        output = 0;
    else
        output = 1;
    end
end

