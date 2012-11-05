Histogram
This application is ported from CUDA SDK sample.

-Overview:
Histogram is commonly used to generate bar graph of data distribution. This is more often used to analyze image processing and in data mining applications.

-Running sample:
This sample generate statistics of data distribution of a char based input data. The result can be used to generate bar graph.
CUDA sample is more tuned to running on NVidia hardware meaning it optimizes to make use of warp size. But in this sample we are not making any such assumption.
To tune this sample like CUDA for NVidia hardware update below constants
    const unsigned histogram_bin_count = 256; 
    const unsigned log2_thread_size = 5U;
    const unsigned thread_count = 6; 
    const unsigned histogram256_tile_size = thread_count * (1U << log2_thread_size);
    const unsigned histogram256_tile_static_memory = (thread_count * histogram_bin_count);
    const unsigned merge_tile_size = 256;
    const unsigned partial_histogram256_count = (240);

-Hardware requirement:
This sample requires DirectX 11 capable card, if none detected sample will use DirectX 11 Reference Emulator.

-Software requirement:
Install Visual Studio 11 from http://msdn.microsoft.com

-References:
http://en.wikipedia.org/wiki/Histogram
http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/histogram256/doc/histogram.pdf
 
