
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <Eigen/Dense>

#include "InfiniteHorizonGP.hpp"
#include "Matern32model.hpp"

int main(int argc, char *argv[])
{
    /* Check input parameters */
    if (argc==1) {
        std::cerr << "No data file provided." << std::endl;
        return 1;
    }
    
    //
    // Read the data
    //
    std::vector< double > X;
    std::vector< double > Y;

    /* Open data file */
    std::ifstream xfile(argv[1]);
    if (!xfile.is_open()) {
        std::cerr << "Cannot read the input file." << std::endl;
        return 1;
    }

    /* Loop through all lines (x,y) */
    while (!xfile.eof()) {
        double x,y;
        xfile >> x;
        xfile >> y;
        if (!xfile.eof()) {
            X.push_back(x);
            Y.push_back(y);
        }
    }
    xfile.close();
    
    std::cout << "Number of data points: " << X.size() << std::endl;

    //
    // Set up model
    //
    
    // Set up the model
    Matern32model model;
    
    // Set hyperparameters
    model.setMagnSigma2(0.1);
    model.setLengthScale(0.1);
    model.setSigma2(0.1);
    
    //
    // Evaluate
    //

    // Start timing
    clock_t startTime = clock();
    
    // Time step length
    double dt = X[1]-X[0];
    
    // Set up the infinite-horizon GP
    InfiniteHorizonGP gp(dt,model.getF(),model.getH(),model.getPinf(),model.getR(),model.getdF(),model.getdPinf(),model.getdR());
    
    // Loop through data
    for (int k = 0; k < Y.size(); k++) {
      gp.update(Y[k]);
    }

    // End timing
    clock_t endTime = clock();
    double elapsed_secs = double(endTime - startTime) / CLOCKS_PER_SEC;
    
    std::cout << "Elapsed time is " << elapsed_secs << " seconds." << std::endl;
    std::cout << "lik = " << std::endl << gp.getLik() << std::endl;
    std::cout << "deriv = " << std::endl << gp.getLikDeriv() << std::endl;

    return 0;
}
