/* webgpu.h includes <math.h> only for NAN; the native library links no libc. */
#ifndef NAN
#define NAN (__builtin_nanf(""))
#endif
