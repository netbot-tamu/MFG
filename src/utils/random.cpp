/*
 * File:   random.c
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

#include "random.h"
 #include <iostream>

/* XOR Shift 64-bit */

uint64_t x; /* The state must be seeded with a nonzero value. */

uint64_t xorshift64star(void) {
	x ^= x >> 12; // a
	x ^= x << 25; // b
	x ^= x >> 27; // c
	return x * UINT64_C(2685821657736338717);
}

/* XOR Shift 1024-bit */

/* The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed,  we suggest to seed a xorshift64* generator and use its
   output to fill s. */
uint64_t s[ 16 ];
int p;

uint64_t xorshift1024star(void) {
	uint64_t s0 = s[ p ];
	uint64_t s1 = s[ p = ( p + 1 ) & 15 ];
	s1 ^= s1 << 31; // a
	s1 ^= s1 >> 11; // b
	s0 ^= s0 >> 30; // c
	return ( s[ p ] = s0 ^ s1 ) * UINT64_C(1181783497276652981);
}

/*
 * Seed our PRNG by first setting the 64-bit seed value for the 64-bit
 * algorithm, then using that seeded algorithm to generate the 1024-bit seed for
 * the 1024-bit algorithm (the real seed value for our implementation).
 */
void seed_xrand(uint64_t seed) {
   x = seed;
   for (int i = 0; i < 16; i++)
      s[i] = xorshift64star();
}

uint64_t xrand(void) {
   return xorshift1024star();
}
