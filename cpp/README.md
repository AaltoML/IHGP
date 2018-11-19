## Infinite-horizon GPs: C++ implementation

The main purpose of this C++ implementation is to power the iOS example. However, as a reference, we provide some codes in C++. Note that the C++ codes only cover a subset of the methods in the paper (Gaussian likelihoods and only the Matèrn 3/2 covariance function), and for simplicity no external libraries are used for solving the associated discrete Riccati equations (the code implements its own iteration-based solver which is rather inefficient).

### Dependencies

This implementation uses [eigen](http://eigen.tuxfamily.org) for matrix operations. On OS X (if you use Homebrew) the dependency can be installed through

```brew install eigen```

The depency packages are handled through `pkg-config`

```brew install pkg-config```

Apart from that you will need the Apple command line tools (if you got this far, you probably already have them installed).

### Running

An example of how to compile the codes is set up in the `Makefile`. Run by

```make```

The file `main.cpp` implements a smoke test script for the functions. You need to pass it a data file in the arguments when you run it.

### MEX file for Matlab

The C++ implementation can be leveraged in Matlab as well. The code in `ihgpr_mex.cpp` can be compiled (aka mexed) through Matlab to a mex file that can be called as any other Matlab function. See the Matlab script `make_mex.m` for brief instructions how to compile to file. This example evaluates the (negative) marginal likelihood and its gradient.


