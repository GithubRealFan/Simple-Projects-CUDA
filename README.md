# Vector Addition
 
1.	Explain how the program is compiled and run
Install Visual Studio.
Install CUDA Toolkit.
Run Developer Command Prompt for Visual Studio (for cl.exe)
nvcc 1.cu -o 1
1.exe 1024 (1024 is the argument value for 1.exe)
2.	For a vector length of N:
-	How many floating operations are being performed in your vector add kernel?
N
Vector Add Kernel performs single addition operation per pair of values in each vector.
So there will be only N floating operations.
-	How many global memory reads are being performed by your kernel?
N * 2
Vector Add Kernel reads every value in each vector only once.
So the number of global memory reads is the same as the number of values in two vertors.
3.	For a vector length of 1024:
-	Explain how many CUDA threads and thread blocks you used.
1024 threads per block and 1 thread block
I configured the number of threads as 1024.
Then the number of thread blocks will be 1. (which is ceil(vector length / number of threads)).
-	Profile your program with Nvidia Nsight. What Achieved Occupancy did you get?
- Execute the command below.
nvprof --metrics achieved_occupancy 1.exe 1024
- Then you can see the result like below.


Invocations                Metric Name                 Metric Description        Min         Max         Avg
    	 1                 achieved_occupancy          Achieved Occupancy      0.466415    0.466415    0.466415
	 
- Achieved Occupancy values are:
0.466419 (Min), 0.466419 (Max), 0.466419 (Avg)
4.	Now increase the vector length to 131070:
-	Did your program still work? if not, what changes did you make?
The program works successfully without change.
-	Explain how many CUDA threads and thread blocks you used.
1024 threads per block and 128 thread blocks
I configure the number of threads as 1024.
Then the number of thread blocks is 128 = ceil(131070 / 1024).
-	Profile your program with Nvidia Nsight. What Achieved Occupancy do you get now?
- Follow the instruction above.
- The Achieved Occupancy values are:
0.800255 (Min), 0.800255 (Max), 0.800255 (Avg)
5.	Further increase the vector length (try 6-10 different vector length).
We have code part for calculating execution time for each step (Host to Device, Kernel, Device to Host). This function is done using clock() function which returns the current running time in milliseconds.
Run command - 1.exe 1000000. Then you will see such result below:
The input length is 1000000
	Time Elapsed: 7 ms
	Time Elapsed: 0 ms
	Time Elapsed: 4 ms
So the execution time of Host to Device is 7 ms, 0 ms for Kernel, 4 ms for Device to Host.
I’ve run the program with several different values and recorded the execution times. (1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864)
And I plotted the values in bar chart format using python matplotlib.
Kernel execution time is nearly 0 ms. So it looks like there is no bar in the chart.
Here is the bar chart.

![image](https://user-images.githubusercontent.com/121934188/211290291-be9cf1c6-79f6-442d-85e6-567c9325b0c2.png)


# Matrix Multiplication

1.	Name 3 applications domains of matrix multiplication.
Fourier Transform, Digital Video Processing, Basic Linear Algebra Subprograms, Robotics
2.	How many floating operations are being performed in your matrix multiply kernel?
(Let’s denote that matrix dimensions for A, B, C are nARows x nACols, nBRows x nBCols, nCRows x nCCols)
The total number of floating operations = nARows * nACols * nBCols * 2
Of course, we know that nARows * nACols * nBCols multiplications are required to perform matrix multiplication.
Thus, in the kernel, we perform nARows * nACols * nBCols multiplication operations - one multiplication operation per one pair of corresponding values in each matrix.
Then we need to sum the calculated values. There will be one addition operation per one multiplication.
Finally, the total number of floating operations is nARows * nACols * nBCols * 2 (for multiplication & addition).
3.	How many global memory reads are being performed by your kernel?
nARows * nACols * nBCols * 2
In the kernel, we perform nARows * nACols * nBCols multiplication operations.
Each time for multiplication, we read two values in each matrix and perform multiplication.
So the total number of global memory read is as above - twice of the number of multiplication operations.
4.	For a matrix A of (128 x 128) and B of (128 x 128):
-	Explain how many CUDA threads and thread blocks you used.
1024 threads per block and 16 thread blocks
I configured the number of threads per dimension as 32. So the total number of threads is 32 x 32 = 1024.
Then the number of thread blocks is ceil(128 / 32) * (128 / 32) = 16.
-	Profile your program with Nvidia Nsight. What Achieved Occupancy did you get?
  Execute the command below.
nvprof --metrics achieved_occupancy 2.exe 128 128 128
  Then you can see the result like below.
Invocations                      Metric Name          Metric Description         Min             Ma x             Avg
          1                achieved_occupancy          Achieved Occupancy      0.940024    0.940024    0.940024
  Achieved Occupancy values are:
0.940024 (Min), 0.940024 (Max), 0.940024 (Avg)
5.	For a matrix A of (511 x 1023) and B of (1023 x 4094):
-	Did your program still work? If not, what changes did you make?
The program works successfully without change.
-	Explain how many CUDA threads and thread blocks you used.
I configured the number of threads per dimension as 32. So the total number of threads is 32 x 32 = 1024.
And the number of thread blocks is 2048 = ceil(511 / 32) * ceil(4094 / 32).
-	Profile your program with Nvidia Nsight. What Achieved Occupancy do you get now?
  Follow the instruction above.
  The Achieved Occupancy values are:
0.910523 (Min), 0.910523 (Max), 0.910523 (Avg)
6.	Further increase the size of matrix A and B (any amount is OK). Plot a 
  
