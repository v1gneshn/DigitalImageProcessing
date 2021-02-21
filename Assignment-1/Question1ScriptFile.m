%Script file for Question 1
%This code reads a 5x5 matrix and evaluates the following
%    1. sum
%    2. maximum
%    3. mean  
%    4. median
%    5. standard deviation 

%all outputs from the function evaluation are directly stored in the
%workspace under the variable names given

%definition of 5x5 matrix
A = [5,6,5,7,5;2,3,2,4,2;3,4,3,5,3;4,5,4,6,4;1,2,1,3,1];
[sum, max, mean, median, stddev, mode] = Question1Fun(A);

fprintf("Sum: %f\n", sum);
fprintf("Maximum: %f\n", max);
fprintf("Mean: %f\n", mean);
fprintf("Median: %f\n", median);
fprintf("Standard Deviation: %f\n", stddev);
fprintf("Mode: %f\n", mode);