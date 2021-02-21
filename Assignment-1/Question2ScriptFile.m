%Script file for Question 2
%This code takes the size of the matrix as input from the user and creates
%a random matrix of the specified size with values in the interval [1 10] 
%and evaluates all functions listed in Question 1

%all outputs from the function evaluation are directly stored in the
%workspace under the variable names given

prompt = "Please enter number of rows (integer value): ";
nrows = input(prompt);
prompt = "Please enter number of columns (integer value): ";
ncols = input(prompt);

A = randi([0,10],nrows,ncols); %random matrix is created
[sum, max, mean, median, stddev, mode] = Question2Fun(A);
fprintf("Sum: %f\n", sum);
fprintf("Maximum: %f\n", max);
fprintf("Mean: %f\n", mean);
fprintf("Median: %f\n", median);
fprintf("Standard Deviation: %f\n", stddev);
fprintf("Mode: %f\n", mode);
