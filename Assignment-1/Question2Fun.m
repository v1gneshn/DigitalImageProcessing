function [sum, maxVal, mean, median, stddev, mode] = Question2Fun(A)
%Question2Fun calculates the following given any matrix as input:
%    1. sum
%    2. maximum
%    3. mean  
%    4. median   
%    5. standard deviation       

[nrows, ncols] = size(A);
%----Sum-----------------------------------------------------------
sum = 0;
num = 0;
for i = 1:nrows
    for j = 1:ncols
        num = num + 1;
        sum = sum + A(i,j);
    end
end

%----Maximum-------------------------------------------------------
maxVal = A(1,1);
for i = 1:nrows
    for j = 1:ncols
        if A(i,j) > maxVal
            maxVal = A(i,j);
        end
    end
end

%----Mean----------------------------------------------------------
mean = sum/num;

%----Median--------------------------------------------------------
Avec = reshape(A', [], 1);
Avec = sort(Avec);
if mod(num, 2) == 0
    median = (Avec(num/2) + Avec(num/2 + 1))/2;
else
    median = Avec(ceil(num/2));
end

%----Standard Deviation--------------------------------------------
sumsq = 0;
for i = 1:num
    sumsq = sumsq + (Avec(i) - mean)^2;
end
stddev = (sumsq/num)^0.5;

%----Frequency Distribution----------------------------------------
distVals = unique(Avec);
n = numel(distVals);
freqs = zeros(n,1);
for i = 1:n
    for j = 1:num
    if distVals(i)==Avec(j)
        freqs(i) = freqs(i) + 1;
    end
    end
end
freqDist = [distVals,freqs];
fprintf("Frequency Distribution in the form - Value , Number of instances\n");
freqDist

%----Mode----------------------------------------------------------
[maxCount, idx] = max(freqs);
mode = distVals(idx);
            


end

