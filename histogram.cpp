//////////////////////////////////////////////////////////////////////////////
//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// File: Histogram.cpp
// 
// Implements histogram in C++ AMP
// Refer README.txt
//----------------------------------------------------------------------------

#include "histogram.h"
#include "CStopWatch.h"
#include <assert.h>
#include <iostream>

#pragma warning (disable : 4267)

#define USE_RGBA
#ifdef USE_RGBA
#define GET_A(COLOR) (( COLOR >> 24) & 0xFF)
#define GET_B(COLOR) (( COLOR >> 16) & 0xFF)
#define GET_G(COLOR) (( COLOR >> 8 ) & 0xFF)
#define GET_R(COLOR) (  COLOR        & 0xFF)
#define RGBA(R, G, B, A) ((A << 24) | (B << 16) | (G << 8) | R);
#else
#define GET_A(COLOR) (( COLOR >> 24) & 0xFF)
#define GET_R(COLOR) (( COLOR >> 16) & 0xFF)
#define GET_G(COLOR) (( COLOR >> 8 ) & 0xFF)
#define GET_B(COLOR) (  COLOR        & 0xFF)
#define RGBA(R, G, B, A) ((A << 24) | (R << 16) | (G << 8) | B);
#endif


const unsigned histogram_bin_count = 256; // Bin count

const unsigned log2_thread_size = 5U;
const unsigned thread_count = 8; // number of partial histogram per tile

const unsigned histogram256_tile_size = (thread_count * (1U << log2_thread_size));
const unsigned histogram256_tile_static_memory = (thread_count * histogram_bin_count);

const unsigned merge_tile_size = histogram_bin_count; // Partial result Merge size
const unsigned partial_histogram256_count = (thread_count * (1U << log2_thread_size));

// Atomically update bucket count
// This function decodes packed byte size data to identify bucket
void histogram_amp::addword_256(unsigned *s_hist_r, unsigned *s_hist_g, unsigned *s_hist_b, unsigned offset, unsigned data) restrict (amp)
{
        unsigned int a = GET_A(data);
        unsigned int r = GET_R(data);
        unsigned int g = GET_G(data);
        unsigned int b = GET_B(data);

        atomic_fetch_inc(&(s_hist_r[offset + r]));
        atomic_fetch_inc(&(s_hist_g[offset + g]));
        atomic_fetch_inc(&(s_hist_b[offset + b]));

//    atomic_fetch_add(&(s_hist[offset + (data >>  0) & 0xFFU]), 1);
//    atomic_fetch_add(&(s_hist[offset + (data >>  8) & 0xFFU]), 1);
//    atomic_fetch_add(&(s_hist[offset + (data >> 16) & 0xFFU]), 1);
//    atomic_fetch_add(&(s_hist[offset + (data >> 24) & 0xFFU]), 1);
}

// This functions divides data among _no_of_tiles*histogram256_tile_size threads.
// And in each tile, threads are grouped by (1 << log2_thread_size) number of threads to update 
// partial histogram count.
template <int _no_of_tiles>
void histogram_amp::histo_kernel(unsigned data_in_uint_count, array<unsigned, 1>& data, array<unsigned, 1>& partial_result_r, array<unsigned, 1>& partial_result_g, array<unsigned, 1>& partial_result_b)
{
    assert((histogram256_tile_size % (1 << log2_thread_size)) == 0, "Threads in a tile should be grouped equally among bin count");
    assert(histogram256_tile_static_memory == (thread_count * histogram_bin_count), "Shared memory size should be in multiple of bin count and number of threads per tile");

    extent<1> e_compute(_no_of_tiles*histogram256_tile_size);
    parallel_for_each(e_compute.tile<histogram256_tile_size>(), 
        [=, &data, &partial_result_r, &partial_result_g, &partial_result_b] (tiled_index<histogram256_tile_size> tidx) restrict(amp)
        {
            tile_static unsigned s_hist_r[histogram256_tile_static_memory];
            tile_static unsigned s_hist_g[histogram256_tile_static_memory];
            tile_static unsigned s_hist_b[histogram256_tile_static_memory];

            // initialize shared memory - each thread will ZERO 8 locations
            // 24, actually... 8 in each of the R, G, and B histograms
            unsigned per_thread_init = histogram256_tile_static_memory / histogram256_tile_size;
            for(unsigned i = 0; i < per_thread_init; i++)
            {
                s_hist_r[tidx.local[0] + i * histogram256_tile_size] = 0;
                s_hist_g[tidx.local[0] + i * histogram256_tile_size] = 0;
                s_hist_b[tidx.local[0] + i * histogram256_tile_size] = 0;
            }
            tidx.barrier.wait();

            // Each group of 32 threads(tidx.local[0] >> log2_thread_size), will update bin count in shared memory
            unsigned offset = (tidx.local[0] >> log2_thread_size) * histogram_bin_count;

            // Entire data is divided for processing between _no_of_tiles*histogram256_tile_size threads
            // There are totally _no_of_tiles tiles with histogram256_tile_size thread in each of them - this is the increment size
            // Consecutive threads do take advantage of memory coalescing
            for(unsigned pos = tidx.global[0]; pos < data_in_uint_count; pos += histogram256_tile_size * partial_histogram256_count)
            {
                unsigned datum = data[pos];
                addword_256(s_hist_r, s_hist_g, s_hist_b, offset, datum);
            }
            tidx.barrier.wait();

            for(unsigned bin = tidx.local[0]; bin < histogram_bin_count; bin += histogram256_tile_size)
            {
                unsigned sumR = 0;
                unsigned sumG = 0;
                unsigned sumB = 0;
                // Updated each threads partial sum to partial histogram
                for(unsigned i = 0; i < thread_count; i++)
                {
                    sumR += s_hist_r[bin + i * histogram_bin_count];
                    sumG += s_hist_g[bin + i * histogram_bin_count];
                    sumB += s_hist_b[bin + i * histogram_bin_count];
                }
                partial_result_r[tidx.tile[0] * histogram_bin_count + bin] = sumR;
                partial_result_g[tidx.tile[0] * histogram_bin_count + bin] = sumG;
                partial_result_b[tidx.tile[0] * histogram_bin_count + bin] = sumB;
            }
        });
}

// This function aggregates partial results
template <int _no_of_tiles>
void histogram_amp::histo_merge_kernel(array<unsigned, 1>& partial_result_r, array<unsigned, 1>& partial_result_g, array<unsigned, 1>& partial_result_b, array<unsigned, 1>& histogram_amp_r, array<unsigned, 1>& histogram_amp_g, array<unsigned, 1>& histogram_amp_b, unsigned hist_bin_count)
{
    assert((partial_result_r.get_extent().size() % histogram_bin_count) == 0);
    assert((partial_result_r.get_extent().size() % _no_of_tiles) == 0, "Bound checking partial histogram array size");

    extent<1> e_compute(hist_bin_count*_no_of_tiles);
    parallel_for_each(e_compute.tile<_no_of_tiles>(),
        [=, &partial_result_r, &partial_result_g, &partial_result_b, &histogram_amp_r, &histogram_amp_g, &histogram_amp_b] (tiled_index<_no_of_tiles> tidx) restrict(amp)
        {
            unsigned sumR = 0;
            unsigned sumG = 0;
            unsigned sumB = 0;
            for (unsigned i = tidx.local[0]; i < partial_result_r.get_extent().size(); i += _no_of_tiles)
            {
                sumR += partial_result_r[tidx.tile[0] + i * histogram_bin_count];
                sumG += partial_result_g[tidx.tile[0] + i * histogram_bin_count];
                sumB += partial_result_b[tidx.tile[0] + i * histogram_bin_count];
            }

            tile_static unsigned s_data_r[_no_of_tiles];
            tile_static unsigned s_data_g[_no_of_tiles];
            tile_static unsigned s_data_b[_no_of_tiles];
            s_data_r[tidx.local[0]] = sumR;
            s_data_g[tidx.local[0]] = sumG;
            s_data_b[tidx.local[0]] = sumB;

            // parallel reduce within a tile
            for (int stride = _no_of_tiles / 2; stride > 0; stride >>= 1)
            {
                tidx.barrier.wait();
                if (tidx.local[0] < stride)
                {
                    s_data_r[tidx.local[0]] += s_data_r[tidx.local[0] + stride];
                    s_data_g[tidx.local[0]] += s_data_g[tidx.local[0] + stride];
                    s_data_b[tidx.local[0]] += s_data_b[tidx.local[0] + stride];
                }
            }

            // tile sum is updated to result array by zero-th thread
            if (tidx.local[0] == 0)
            {
                histogram_amp_r[tidx.tile[0]] = s_data_r[0];
                histogram_amp_g[tidx.tile[0]] = s_data_g[0];
                histogram_amp_b[tidx.tile[0]] = s_data_b[0];
            }
        });
}


// This function generates random input data and initializes
// bin count
histogram::histogram(unsigned bin_count, unsigned data_size_in_mb)
{
    data_type_size = sizeof(user_data_t);
    unsigned byte_count = 1200 * 1600 * data_type_size; //data_size_in_mb * (1024 * 1024);
    int item_count =  byte_count / data_type_size;

    // This algorithm is tuned to work for 256 bin count
    assert(histogram_bin_count == bin_count, "Current implementation is tuned for 256 bin count");

    data.resize(item_count);
    histo_amp_r.resize(bin_count);
    histo_amp_g.resize(bin_count);
    histo_amp_b.resize(bin_count);

    assert(item_count > 0, "No data to process");

    // number of relevant data bits in a random number
    //unsigned data_mask = (unsigned)(pow(2, data_type_size * 8) - 1);
    srand(2012);
    for (int i = 0; i < item_count; i++)
        data[i] = RGBA((int)(rand()) % 256, (int)(rand()) % 256, (int)(rand()) % 256, (int)(rand()) % 256);// & data_mask;

    for (unsigned i = 0; i < bin_count; i++)
    {
      histo_amp_r[i] = 0;
      histo_amp_g[i] = 0;
      histo_amp_b[i] = 0;
    }
}

histogram::~histogram()
{

}

// This function creates copies of data on accelerator, dispatch 2 kernels, 
// 1st to calculate histogram partially and then another to merge the results.
// Finally copy result from accelerator to host memory
void histogram::run()
{
    unsigned byte_count = data.size() * data_type_size;
    assert(byte_count % sizeof(unsigned) == 0);
    unsigned data_uint_count = byte_count / sizeof(unsigned);

    const unsigned histo_count = histo_amp_r.size();

    array<unsigned, 1> histogram_r(histo_amp_r.size());
    array<unsigned, 1> histogram_g(histo_amp_g.size());
    array<unsigned, 1> histogram_b(histo_amp_b.size());
    // interpret all data as unsigned
    array<unsigned, 1> a_data(byte_count / sizeof(unsigned), (unsigned*)data.data());
    array<unsigned, 1> partial_r(partial_histogram256_count * histo_amp_r.size());
    array<unsigned, 1> partial_g(partial_histogram256_count * histo_amp_g.size());
    array<unsigned, 1> partial_b(partial_histogram256_count * histo_amp_b.size());

    CStopWatch gpu;
    gpu.startTimer();
    // Run kernels to generate partial histogram - stage 1
    histogram_amp::histo_kernel<partial_histogram256_count>(data_uint_count, a_data, partial_r, partial_g, partial_b);
    // Run kernel to merge partial histograms - stage 2
    histogram_amp::histo_merge_kernel<merge_tile_size>(partial_r, partial_g, partial_b, histogram_r, histogram_g, histogram_b, histo_amp_r.size());
    histogram_r.accelerator_view.wait();
    histogram_g.accelerator_view.wait();
    histogram_b.accelerator_view.wait();
    gpu.stopTimer();

    std::cout << "GPU Time: " << gpu.getElapsedTime() * 1000.0 << " ms." << std::endl;
    copy(histogram_r, histo_amp_r.begin());
    copy(histogram_g, histo_amp_g.begin());
    copy(histogram_b, histo_amp_b.begin());
}

// This function computes histogram on CPU (as a baseline) and compare with 
// GPU results. If there is any error printf out the first mismatched result
bool histogram::verify()
{
    std::vector<unsigned> histo_cpu_r(histo_amp_r.size());
    std::vector<unsigned> histo_cpu_g(histo_amp_g.size());
    std::vector<unsigned> histo_cpu_b(histo_amp_b.size());

    for (unsigned i = 0; i < histo_cpu_r.size(); i++)
    {
        histo_cpu_r[i] = 0;
        histo_cpu_g[i] = 0;
        histo_cpu_b[i] = 0;
    }

    // Number of higher order bits used to calculate bucket
    //unsigned data_shift = (data_type_size * 8) - log2(histo_amp.size());

    CStopWatch cpu;
    cpu.startTimer();
    // histogram on cpu
    for (unsigned i = 0; i < data.size(); i++)
    {
        unsigned color = data[i];
        unsigned int a = GET_A(color);
        unsigned int r = GET_R(color);
        unsigned int g = GET_G(color);
        unsigned int b = GET_B(color);

        histo_cpu_r[r]++;
        histo_cpu_g[g]++;
        histo_cpu_b[b]++;
    }
    cpu.stopTimer();
    std::cout << "CPU Time: " << cpu.getElapsedTime() * 1000.0 << " ms." << std::endl;

    unsigned int cpuSumR = 0;
    unsigned int cpuSumG = 0;
    unsigned int cpuSumB = 0;
    unsigned int gpuSumR = 0;
    unsigned int gpuSumG = 0;
    unsigned int gpuSumB = 0;
    // verify data
    for (unsigned i = 0; i < histo_cpu_r.size(); i++)
    {
#if 0
      std::cout << "R: INDEX " << std::hex << (i % 256) << " CPU: " << std::dec << histo_cpu_r[i] << " GPU: " << std::dec << histo_amp_r[i] << std::endl;
      std::cout << "G: INDEX " << std::hex << (i % 256) << " CPU: " << std::dec << histo_cpu_g[i] << " GPU: " << std::dec << histo_amp_g[i] << std::endl;
      std::cout << "B: INDEX " << std::hex << (i % 256) << " CPU: " << std::dec << histo_cpu_b[i] << " GPU: " << std::dec << histo_amp_b[i] << std::endl;
      cpuSumR += histo_cpu_r[i];
      cpuSumG += histo_cpu_g[i];
      cpuSumB += histo_cpu_b[i];

      gpuSumR += histo_amp_r[i];
      gpuSumG += histo_amp_g[i];
      gpuSumB += histo_amp_b[i];
#endif
        if (histo_amp_r[i] != histo_cpu_r[i])
        {
            std::cout << i << ": " << histo_amp_r[i] << " <> " << histo_cpu_r[i] << std::endl;
            std::cout << "***HISTOGRAM " << histo_amp_r.size() << " VERIFICATION FAILURE***" << std::endl;
            return false;
        }
        if (histo_amp_g[i] != histo_cpu_g[i])
        {
            std::cout << i << ": " << histo_amp_g[i] << " <> " << histo_cpu_g[i] << std::endl;
            std::cout << "***HISTOGRAM " << histo_amp_g.size() << " VERIFICATION FAILURE***" << std::endl;
            return false;
        }
        if (histo_amp_b[i] != histo_cpu_b[i])
        {
            std::cout << i << ": " << histo_amp_b[i] << " <> " << histo_cpu_b[i] << std::endl;
            std::cout << "***HISTOGRAM " << histo_amp_b.size() << " VERIFICATION FAILURE***" << std::endl;
            return false;
        }
    }
#if 0
    std::cout << "cpuSumR = " << cpuSumR << ", gpuSumR = " << gpuSumR << std::endl;
    std::cout << "cpuSumG = " << cpuSumG << ", gpuSumG = " << gpuSumG << std::endl;
    std::cout << "cpuSumB = " << cpuSumB << ", gpuSumB = " << gpuSumB << std::endl;
#endif
    return true;
}

int main()
{
    accelerator default_device;
    std::wcout << L"Using device : " << default_device.get_description() << std::endl;
    if (default_device == accelerator(accelerator::direct3d_ref))
        std::cout << "WARNING!! Running on very slow emulator! Only use this accelerator for debugging." << std::endl;

    // With larger data size there will be better performance, use data size just 
    // enough to amortize copy cost. 
    // For simplicity using 3MB of data
    const int data_size_in_mb = 3;

    histogram _histo_char(histogram_bin_count, data_size_in_mb);
    _histo_char.run();
    _histo_char.verify();
}

