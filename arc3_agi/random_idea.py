from numpy import packbits, random, uint8, uint64, zeros

# In a uniform random 64 bit string each bit has a 50:50 chance of being a 1 or a 0.
# By bitwise ANDing two random 64 bit strings together, we can create a new random 64 bit string that
# has a 25% chance of being a 1 and a 75% chance of being a 0. We can do this recursively to create a
# random 64 bit string that has a 1 in 2^n chance of being a 1.
# We can also take the bitwise OR of two random 64 bit strings to create a new random 64 bit string
# that has a 75% chance of being a 1 and a 25% chance of being a 0. By mixing and matching these two
# operations, we can create random 64 bit string that have almost any desired probability of being a
# 1 or a 0.
#
# However it is expensive to generate a new random 64 bit string for each operation and do all
# the bitwise operations. The goal here is to validate mechanism for generating random 64 bit strings
# with a desired probability of being a 1 or a 0, and to see if we can do it more efficiently. The
# Essential premise is to reuse random strings by caching them and shuffling them around. If we use
# 256 element tables we can directly index into them with 8 bits of a random number to get a new
# random number with the desired probability.
