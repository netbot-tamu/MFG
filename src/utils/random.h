/*
 * File:   random.h
 * Author: madtreat
 *
 * Created on December 17, 2014, 4:12 PM
 *
 * Our implementation of 2 Psuedo-Random Number Generators, both variants of the
 * XOR Shift PRNG algorithm.  Both variants here were copied directly from
 * Wikipedia (http://en.wikipedia.org/wiki/Xorshift), which copied them directly
 * from "An experimental exploration of Marsaglia's xorshift generators,
 * scrambled" by Sebastiano Vigna (2014), located online at
 * http://arxiv.org/abs/1402.6246
 */

#ifndef RANDOM_H
#define RANDOM_H

#ifdef	__cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
typedef unsigned __int64 uint64_t;
#define UINT64_C(val) (val##ui64)
#else // _MSC_VER
#include <stdint.h>
// if UINT64_C is already defined in stdint.h, there is no need to redefine it
#ifndef UINT64_C
#define UINT64_C(val) (val##ULL)
#endif // UINT64_C

#endif // _MSC_VER

/*
 * This 64-bit XOR Shift* algorithm is used to provide a single 64-bit seed for
 * the 1024-bit XOR Shift* algorithm, which passes more tests than this variant.
 */
uint64_t xorshift64star(void);

/*
 * This 1024-bit XOR Shift* algorithm is our implementation of a PRNG to provide
 * repeatable results from MFG calculations.
 */
uint64_t xorshift1024star(void);

/*
 * A seed function and simple function wrapper for xorshift1024star().
 */
void seed_xrand(uint64_t val);
uint64_t xrand(void);

#ifdef	__cplusplus
}
#endif

#endif	/* RANDOM_H */

