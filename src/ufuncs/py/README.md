# Ufuncs

This is the implementation of [universal functions](https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html) for the square root operation. As the resource suggests this is one of the easiest extensions, and you can get with a couple of lines the following properties on your ufunc:

- broadcasting
- N-dimensional looping
- automatic type-conversions with minimal memory usage
- optional output arrays

## Square Root
I came to learn that this can be an operation done directly by the processor itself. There are two implementations

1. sqrt(x)
2. sqrt2(x)

Both of them make use of the continuous fraction approximation. This iterative process requires different ITERATIONS values dependent on x to reach reasonable accuracy. Easy to see that 

<p align="center">
$$
    \color{orange}{
        \sqrt{x} = a + \frac{x-a^2}{a+\sqrt{x}} 
    }
$$
<p>

The only between the two implementations is the presence of a TOLERANCE cut condition, being  $|last - current| < TOLERANCE $. 


## Tests

Check for yourself, it seems that tolerance present on sqrt2 adds a reasonable amount of overhead to the computation. But the error is very low for a ridicoulous choice of ITERATIONS, e.g. 5 or 10. 

Whenever the operation is done for a high dimensional, or big array, `numpy` wins. But there is a sweet spot to win some femtoseconds :P.

<!-- insert image in assets -->
<img src="tests/assets/ufunc_vs_np.png" alt="sqrt2" width="80%" height="80%">


## To try out

- make them accept `a` as parameter as well
- create `sqrt3` using native `C sqrt`


