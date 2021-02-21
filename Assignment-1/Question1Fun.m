function [sum, maxVal, mean, median, stddev, mode] = Question1Fun(A)
%Question1Fun calculates the following given a 5x5 matrix as input:
%    1. sum
%    2. maximum
%    3. mean  
%    4. median   
%    5. standard deviation       

%----Sum-----------------------------------------------------------
sum = 0;
for i = 1:5
    for j = 1:5
        sum = sum + A(i,j);
    end
end

%----Maximum-------------------------------------------------------
maxVal = A(1,1);
for i = 1:5
    for j = 1:5
        if A(i,j) > maxVal
            maxVal = A(i,j);
        end
    end
end

%----Mean----------------------------------------------------------
mean = sum/25;

%----Median--------------------------------------------------------
Avec = reshape(A', [], 1);
Avec = sort(Avec);
median = Avec(13);

%----Standard Deviation--------------------------------------------
sumsq = 0;
for i = 1:25
    sumsq = sumsq + (Avec(i) - mean)^2;
end
stddev = (sumsq/25)^0.5;

%----Frequency Distribution----------------------------------------
distVals = unique(Avec);
n = numel(distVals);
freqs = zeros(n,1);
for i = 1:n
    for j = 1:25
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
