#include <stdint.h>
#include <stdlib.h>
#include <array>
#include <vector>
#include <zfp.hpp>

namespace zfp_iface {

template<typename T>
std::vector<uint8_t> compress(T* array, size_t arraySize, size_t& compressedSize) {
   // Allocate memory for compressed data

   zfp_stream* zfp = zfp_stream_open(NULL);
   zfp_field* field;

   if (std::is_same<T, float>::value) field = zfp_field_1d(array, zfp_type_float, arraySize);
   if (std::is_same<T, double>::value) field = zfp_field_1d(array, zfp_type_double, arraySize);

   size_t maxSize = zfp_stream_maximum_size(zfp, field);
   std::vector<uint8_t> compressedData(maxSize);

   // Initialize ZFP compression
   zfp_stream_set_accuracy(zfp, 1e-5);
   bitstream* stream = stream_open(compressedData.data(), compressedSize);
   zfp_stream_set_bit_stream(zfp, stream);
   zfp_stream_rewind(zfp);

   // Compress the array
   compressedSize = zfp_compress(zfp, field);
   compressedData.erase(compressedData.begin() + compressedSize, compressedData.end());
   zfp_field_free(field);
   zfp_stream_close(zfp);
   stream_close(stream);
   return compressedData;
}

template<typename T>
std::vector<T> decompressArrayFloat(uint8_t* compressedData, size_t compressedSize, size_t arraySize) {

   // Allocate memory for decompresseFloatd data
   std::vector<T> decompressedArray(arraySize);

   // Initialize ZFP decompression
   zfp_stream* zfp = zfp_stream_open(NULL);
   zfp_stream_set_accuracy(zfp, 1e-5);
   bitstream* stream_decompress = stream_open(compressedData, compressedSize);
   zfp_stream_set_bit_stream(zfp, stream_decompress);
   zfp_stream_rewind(zfp);

   // Decompress the array
   zfp_field* field_decompress;

   if (std::is_same<T, float>::value) field_decompress = zfp_field_1d(decompressedArray.data(), zfp_type_float, decompressedArray.size());
   if (std::is_same<T, double>::value) field_decompress = zfp_field_1d(decompressedArray.data(), zfp_type_double, decompressedArray.size());

   size_t retval = zfp_decompress(zfp, field_decompress);
   (void)retval;
   zfp_field_free(field_decompress);
   zfp_stream_close(zfp);
   stream_close(stream_decompress);

   return decompressedArray;
}

}
