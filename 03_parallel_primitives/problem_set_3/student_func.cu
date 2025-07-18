/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
#include <cuda.h>
#include "utils.h"

__device__ float _min(float a, float b)
{
	return a > b ? b : a;
}
__device__ float _max(float a, float b)
{
	return a > b ? a : b;
}

__global__ void minmax_reduce_global(float* dout, float* din, unsigned int n,
	bool isMin)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			din[idx] =
				isMin ? _min(din[idx], din[idx + s]) : _max(din[idx], din[idx + s]);
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		dout[blockIdx.x] = din[idx];
	}
}

__global__ void minmax_reduce_shared(float* dout, const float* din, unsigned int n,
	bool isMin)
{
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int global_id = tid + blockDim.x * blockIdx.x;
	if (global_id >= n)
		sdata[tid] = 0;
	else
		sdata[tid] = din[global_id];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sdata[tid] = isMin ? _min(sdata[tid], sdata[tid + s])
							   : _max(sdata[tid], sdata[tid + s]);
		__syncthreads();
	}

	if (tid == 0)
	{
		dout[blockIdx.x] = sdata[0];
	}
}

__global__ void histo_atmoic(unsigned int* out_histo, const float* d_in,
	int numBins, int input_size, float minval,
	float range)
{
	int tid = threadIdx.x;
	int global_idx = tid + blockDim.x * blockIdx.x;
	if (global_idx >= input_size)
		return;
	int bin = ((d_in[global_idx] - minval) * numBins) / range;
	bin = bin == numBins ? numBins - 1 : bin;
	atomicAdd(&out_histo[global_idx], 1);
}

//--------HILLIS-STEELE SCAN----------
// Optimal step efficiency (histogram is a relatively small vector)

__global__ void scan_hills_steele(unsigned int* d_out, const unsigned* d_in, int size)
{
	extern __shared__ unsigned int temp[];

	int tid = threadIdx.x;
	int pout = 0, pin = 1;
	temp[tid] = tid > 0 ? d_in[tid - 1] : 0; // exclusive scan
	__syncthreads();

	// double buffered
	for (int off = 1; off < size; off <<= 1)
	{
		pout = 1 - pout;
		pin = 1 - pout;
		if (tid >= off)
			temp[size * pout + tid] = temp[size * pin + tid] + temp[size * pin + tid - off];
		else
			temp[size * pout + tid] = temp[size * pin + tid];
		__syncthreads();
	}
	d_out[tid] = temp[pout * size + tid];
}

float reduce(const float* d_logLuminance, int input_size, bool isMin)
{
	int THREAD_PER_BLOCK = 32;
	int numBlOCKS = (input_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

	// allocate memory for intermediate results
	float* d_out;
	checkCudaErrors(cudaMalloc(&d_out, numBlOCKS * sizeof(float)));
	minmax_reduce_shared<<<numBlOCKS, THREAD_PER_BLOCK, THREAD_PER_BLOCK * sizeof(float)>>>(d_out, d_logLuminance, input_size, isMin);
	cudaDeviceSynchronize();

	THREAD_PER_BLOCK = numBlOCKS;
	numBlOCKS = 1;
	float results;
	minmax_reduce_shared<<<numBlOCKS, THREAD_PER_BLOCK, THREAD_PER_BLOCK * sizeof(float)>>>(&results, d_out, input_size, isMin);

	return results;
}

unsigned int* compute_hisograme(const float* const d_logLuminance, int numBins, int input_size, float minVal, float range)
{
	unsigned int* hist_out;
	checkCudaErrors(cudaMalloc(&hist_out, sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(hist_out, 0, numBins * sizeof(unsigned int)));
	int THREAD_PER_BLOCK = 32;
	int BLOCKS = (input_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	histo_atmoic<<<BLOCKS, THREAD_PER_BLOCK>>>(hist_out, d_logLuminance, numBins, input_size, minVal, range);
	cudaDeviceSynchronize();
	return hist_out;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf, float& min_logLum,
	float& max_logLum, const size_t numRows,
	const size_t numCols, const size_t numBins)
{
	// TODO
	/*Here are the steps you need to implement
	  1) find the minimum and maximum value in the input logLuminance channel
		 store in min_logLum and max_logLum*/
	// reduce
	int n = numRows * numCols;
	min_logLum = reduce(d_logLuminance, n, true);
	max_logLum = reduce(d_logLuminance, n, false);

	// 2) subtract them to find the range
	float range = max_logLum - min_logLum;

	/*3) generate a histogram of all the values in the logLuminance channel using
	   the formula: bin = (lum[i] - lumMin) / lumRange * numBins*/

	unsigned int* d_hist = compute_hisograme(d_logLuminance, numBins, n, min_logLum, range);

	/*4) Perform an exclusive scan (prefix sum) on the histogram to get
	   the cumulative distribution of luminance values (this should go in the
	   incoming d_cdf pointer which already has been allocated for you)       */
	scan_hills_steele<<<1, numBins, 2 * numBins * sizeof(unsigned int)>>>(d_cdf, d_hist, numBins);
}
