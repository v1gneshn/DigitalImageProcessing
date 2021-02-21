# Digital Image Processing Assignment-1

## Group Members

- Vignesh Nagarajan(EDM18B055)
- Aditya D.S. (ESD18I001)
- Aditya (MDM)

## Questions

### Q1. Read a matrix of size 5*5 and find the following by using a user-defined function

- sum
- maximum
- mean
- median
- mode
- standard deviation
- frequency distribution

#### Code

```c
// Question1Fun.m

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

```

```c
// Question1ScriptFile.m

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
```

#### Outputs

| ![Matrix][Q1-matrix] | ![Output][Q1-output] |
|:--------------------------:|:--------------------------:|

### Q2. Read the matrix size through the keyboard and create a random matrix of integers ranging from  0 to 10 and compute all the above functions listed in question 1

#### Code

```c
// Question2Fun.m

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


```

```c
// Question2ScriptFile.m

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

```

#### Outputs

| ![Matrix][Q2-matrix] | ![Output][Q2-output] |
|:--------------------------:|:--------------------------:|

### Q3. Take a Lena image and convert it into grayscale. Add three different types of noises(salt and pepper, additive Gaussian noise, speckle), each noise in the sets of 5,10,15,20,25,30. Take average for each set and display the average images. Report the observation made

#### Code

```py
import cv2 as cv
from skimage import io
import random
import numpy as np
from matplotlib import pyplot as plt

def noisy(noise_typ,image,prob=0.05):
        
        if noise_typ == "s&p":
            output = np.zeros(image.shape,np.uint8)
            thres = 1 - 0.05 
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j] = 0
                    elif rdn > thres:
                        output[i][j] = 255
                    else:
                        output[i][j] = image[i][j]
            return output
        
        elif noise_typ == "gaussian":
            row,col= image.shape
            mean = 40
            var = 25
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col))
            gauss = gauss.reshape(row,col)
            noisy = image + gauss
            noisy.astype('uint8')
            return noisy
        
        elif noise_typ =="speckle":
            row,col = image.shape
            gauss = np.random.randn(row,col)
            gauss = gauss.reshape(row,col)        
            noisy = image + image * gauss
            return noisy

```

```py
#Qn 3-Test with Salt & Pepper Noise
lena = cv.imread('lena512color.tiff')
lenag=cv.cvtColor(lena,cv.COLOR_BGR2GRAY)
cv.imshow("Lena(Original)",lena)
cv.imshow("Lena(Greyscale)",lenag)
cv.waitKey(0)
cv.destroyAllWindows()



outputimg=[]
for i in range(30):
    outputimg.append(noisy("s&p",lenag))
    
random5imgs=random.sample(outputimg, 5)
avg_img_5 = np.mean(random5imgs, axis=0,dtype=np.int64)
cv.imwrite("Testdamacha_qn3.png",avg_img_5)
(fig, axs) =  plt.subplots(nrows=3, ncols=2, figsize=(40, 40))
fig.suptitle("Averages for Salt & Pepper Noise",size=20)
setlen = [5,10,15,20,25,30]
c=0;
for i in range(3):
    for j in range(2):
        avg_img = np.mean(random.sample(outputimg, setlen[c]), axis=0,dtype=np.int64) 
        axs[i,j].imshow(avg_img,cmap='gray')
        axs[i,j].set_title("{} images avg".format(setlen[c]))  
        axs[i,j].axes.xaxis.set_visible(False)
        axs[i,j].axes.yaxis.set_visible(False)

        c=c+1;

plt.show()
```

```py
#Qn 3-Test with Additive Gaussian Noise
outputimg=[]
for i in range(30):
    noise_gauss = noisy("gaussian", lenag)
    noise_gauss = noise_gauss.astype('uint8')
    outputimg.append(noise_gauss)
    
#cv.imwrite("Testdamacha_qn3.png",avg_img_5)
(fig, axs) =  plt.subplots(nrows=3, ncols=2, figsize=(40, 40))
fig.suptitle("Averages for Gaussian Noise",size=20)
setlen = [5,10,15,20,25,30]
c=0;
for i in range(3):
    for j in range(2):
        avg_img = np.mean(random.sample(outputimg, setlen[c]), axis=0,dtype=np.int64) 
        axs[i,j].imshow(avg_img,cmap='gray')
        axs[i,j].set_title("{} images avg".format(setlen[c]))  
        axs[i,j].axes.xaxis.set_visible(False)
        axs[i,j].axes.yaxis.set_visible(False)

        c=c+1;

plt.show()
```

```py
#Qn 3-Test with Speckle Noise
outputimg=[]
for i in range(30):
    noise_gauss = noisy("speckle", lenag)
    noise_gauss = noise_gauss.astype('uint8')
    outputimg.append(noise_gauss)
    
#cv.imwrite("Testdamacha_qn3.png",avg_img_5)
(fig, axs) =  plt.subplots(nrows=3, ncols=2, figsize=(40, 40))
fig.suptitle("Averages for Gaussian Noise",size=20)
setlen = [5,10,15,20,25,30]
c=0;
for i in range(3):
    for j in range(2):
        avg_img = np.mean(random.sample(outputimg, setlen[c]), axis=0,dtype=np.int64) 
        axs[i,j].imshow(avg_img,cmap='gray')
        axs[i,j].set_title("{} images avg".format(setlen[c]))  
        axs[i,j].axes.xaxis.set_visible(False)
        axs[i,j].axes.yaxis.set_visible(False)

        c=c+1;

plt.show()
```

#### Outputs

| ![Greyscale Image][Q3-1] | ![Salt and Pepper Noise][Q3-2] |
|:--------------------------:|:--------------------------:|

| ![Speckle Noise][Q3-3] | ![AWGN Noise][Q3-4] |
|:--------------------------:|:--------------------------:|

For Salt & Pepper Noise and Speckle noise, the noise levels on the averaged images seems to decrease as we increase the set size from 5 to 30 in steps of 5.

For Additive Gaussian Noise, the noise levels on the averaged images DOES NOT decrease as we increase the samples set size.

### Q4. Download Lena image and scale it by factors of 1,2,0.5 using bilinear interpolation and display the scaled images. Also, display the output of built-in functions for doing scaling by factors of 0.5,2. Compare the results

#### Code

```py
# Q4.py
```

#### Outputs

0.5x and 1x resize
| ![Matrix][Q4-0.5x] | ![Output][Q4-1x] |
|:--------------------------:|:--------------------------:|

2x resize
| ![Matrix][Q4-2x] |
|:--------------------------:|

0.5x and 2x resize using built in command `cv.resize`. The second image is overlapping as there is'nt enough space on the screen.
| ![Matrix][Q4-0.5x-resize] | ![Output][Q4-2x-resize] |
|:--------------------------:|:--------------------------:|

As can be seen, exact same results have been achieved using both the inbuilt command and the function we wrote.

### Q5. Download the leaning tower of PISA image and find the angle of inclination using appropriate rotations with bilinear interpolation

#### Code

#### Outputs

### Q6. Do histogram equalization on pout-dark and display the same

#### Code

```py
#Qn 6 (USING INBUILT FUNCTION)
from matplotlib import pyplot as plt
import cv2 as cv
from skimage import exposure

src = cv.imread('pout-dark.jpg',0)
if src is None:
    print('Could not open or find the image:','pout-dark.jpg')
    exit(0)
dst = cv.equalizeHist(src) 

(fig, axs) =  plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
for (i, image) in enumerate((src, dst)):
        (hist, bins) = exposure.histogram(image,source_range="dtype")
        axs[i,1].plot(bins, hist / hist.max())

axs[0,1].set_title("Source hist(Pout-Dark)");
axs[1,1].set_title("Equalized hist");
fig.suptitle('Histogram Equalization using Inbuilt function',size=16)

axs[0,0].imshow(src,cmap='gray')
axs[1,0].imshow(dst,cmap='gray')

for i in range(2) :
    axs[i,0].axes.xaxis.set_visible(False)
    axs[i,0].axes.yaxis.set_visible(False)

fig.tight_layout()

#plt.show()
axs[0,0].set_title("Pout-Dark");
axs[1,0].set_title("Equalized Image");

```

```py
#Qn 6 using USER-DEFINED function
import numpy as np
from skimage import exposure
import cv2 as cv
from matplotlib import pyplot as plt
def histequalizer_userdefined (src) :     
        img = np.copy(src);
        a = np.zeros((256,),dtype=np.float16)
        
        height,width=img.shape

        #finding histogram
        for i in range(width):
            for j in range(height):
                g = img[j,i]
                a[g] = a[g]+1
            
        #print(a)
        
        #performing histogram equalization
        tmp = 1.0/(height*width)
        b = np.zeros((256,),dtype=np.float16)

        for i in range(256):
            for j in range(i+1):
                b[i] += a[j] * tmp;
            b[i] = round(b[i] * 255);

        # b now contains the equalized histogram
        b=b.astype(np.uint8)

        #Re-map values from equalized histogram into the image
        for i in range(width):
            for j in range(height):
                g = img[j,i]
                img[j,i]= b[g]
                
        (fig, axs) =  plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
        for (i, image) in enumerate((src, dst)):
                (hist, bins) = exposure.histogram(image,source_range="dtype")
                axs[i,1].plot(bins, hist / hist.max())

        axs[0,1].set_title("Source hist(Pout-Dark)");
        axs[1,1].set_title("Equalized hist");
        fig.suptitle('Histogram Equalization(User-defined)',size=16)

        axs[0,0].imshow(src,cmap='gray')
        axs[1,0].imshow(dst,cmap='gray')

        for i in range(2) :
            axs[i,0].axes.xaxis.set_visible(False)
            axs[i,0].axes.yaxis.set_visible(False)

        fig.tight_layout()

        #plt.show()
        axs[0,0].set_title("Pout-Dark");
        axs[1,0].set_title("Equalized Image");  

        
##Feed the image array into the user defined function
src = cv.imread('pout-dark.jpg',0)
if src is None:
    print('Could not open or find the image:','pout-dark.jpg')
    exit(0)

histequalizer_userdefined(src)
```

#### Outputs

| ![Matrix][Q6-1] | ![Output][Q6-2] |
|:--------------------------:|:--------------------------:|

### Q7. Do histogram matching(specification) on the pout-dark image, keeping pout-bright as a reference image

#### Code

```py
from skimage import exposure
import cv2 as cv


ref = cv.imread('pout-bright.jpg',0)
print("[INFO] performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)
# show the output images
cv.imshow("Source", src)
cv.imshow("Reference", ref)
cv.imshow("Matched", matched)
cv.waitKey(0)
cv.destroyAllWindows()

(fig, axs) =  plt.subplots(nrows=3, ncols=2, figsize=(9, 9))
# loop over our source image, reference image, and output matched
# image
for (i, image) in enumerate((src, ref, matched)):
     # convert the image from BGR to RGB channel ordering
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # loop over the names of the channels in RGB order
        # compute a histogram for the current channel and plot it
        (hist, bins) = exposure.histogram(image,source_range="dtype")
        axs[i,1].plot(bins, hist / hist.max())
        # compute the cumulative distribution function for the
        # current channel and plot it
        (cdf, bins) = exposure.cumulative_distribution(image)
        axs[i,1].plot(bins, cdf)
        # set the y-axis label of the current plot to be the name
        # of the current color channel
        #axs[j, 0].set_ylabel(color)

axs[0,1].set_title("Source hist(P-Dark)")
axs[1,1].set_title("Reference hist(P-Bright)")
axs[2,1].set_title("Matched hist");



axs[0,0].imshow(src,cmap='gray')
axs[1,0].imshow(ref,cmap='gray')
axs[2,0].imshow(matched,cmap='gray')

for i in range(3) :
    axs[i,0].axes.xaxis.set_visible(False)
    axs[i,0].axes.yaxis.set_visible(False)

fig.tight_layout()

#plt.show()
axs[0,0].set_title("Source image (P-Dark)");
axs[1,0].set_title("Reference Image(P-Bright)");
axs[2,0].set_title("Matched Image");
```

```py
#Qn 7 matched histogram user defined

def find_value_target(val, target_arr):
    key = np.where(target_arr == val)[0]

    if len(key) == 0:
        key = find_value_target(val+1, target_arr)
        if len(key) == 0:
            key = find_value_target(val-1, target_arr)
    vvv = key[0]
    return vvv


def match_histogram(inp_img, hist_input, e_hist_input, e_hist_target, _print=True):
    '''map from e_inp_hist to 'target_hist '''
    en_img = np.zeros_like(inp_img)
    tran_hist = np.zeros_like(e_hist_input)
    for i in range(len(e_hist_input)):
        tran_hist[i] = find_value_target(val=e_hist_input[i], target_arr=e_hist_target)
    print_histogram(tran_hist, name="trans_hist_", title="Transferred Histogram")
    '''enhance image as well:'''
    for x_pixel in range(inp_img.shape[0]):
        for y_pixel in range(inp_img.shape[1]):
            pixel_val = int(inp_img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = tran_hist[pixel_val]
    '''creating new histogram'''
    hist_img, _ = generate_histogram(en_img, print=False, index=3)
    print_img(img=en_img, histo_new=hist_img, histo_old=hist_input, index=str(3), L=L)
```

#### Outputs

| ![Matrix][Q7-1] | ![Output][Q7-2] |
|:--------------------------:|:--------------------------:|

[Q1-matrix]: outputs/Q1Input5x5Matrix.JPG
[Q1-output]: outputs/Q1Output.JPG
[Q2-matrix]: outputs/Q2InputRandomMatrix.JPG
[Q2-output]: outputs/Q2Output.JPG
[Q3-1]: outputs/Q3-1.png
[Q3-2]: outputs/Q3-salt-pepper.jpg
[Q3-3]: outputs/Q3-speckle.jpg
[Q3-4]: outputs/Q3-WGN.jpg
[Q4-0.5x-resize]: outputs/Q4-0.5x-resize.png
[Q4-2x-resize]: outputs/Q4-2x-resize.png
[Q4-0.5x]: outputs/Q4-0.5x.png
[Q4-1x]: outputs/Q4-1x.png
[Q4-2x]: outputs/Q4-2x.png
[Q6-1]: outputs/Q6-1.jpg
[Q6-2]: outputs/Q6-2.jpg
[Q7-1]: outputs/Q7-1.jpg
[Q7-2]: outputs/Q7-2.jpg
