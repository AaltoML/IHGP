#include <Eigen/Core>
#include <vector>
#include <mex.h>
#include "InfiniteHorizonGP.hpp"

using namespace std;
using namespace Eigen;

// These types are used to map Matlab data to Eigen data without any copying
typedef Map<MatrixXd> MexMat;
typedef Map<VectorXd> MexVec;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    // Get input dimensions
    int n      = mxGetM(prhs[0]);              // # Number of observations
    int dim    = mxGetM(prhs[2]);              // # State dimension
    int nparam = mxGetDimensions(prhs[6])[2];  // # Number of parameters

    // Map inputs (y,dt,F,H,Pinf,R,dF,dQc,dR,dPinf)
    MexMat y     (mxGetPr(prhs[0]), n, 1);
    double dt    =mxGetScalar(prhs[1]);
    MexMat F     (mxGetPr(prhs[2]),  dim, dim);
    MexMat H     (mxGetPr(prhs[3]),  1, dim);
    MexMat Pinf  (mxGetPr(prhs[4]),  dim, dim);
    double R     =mxGetScalar(prhs[5]);

    // Map dF, dPinf, dR
    vector< MatrixXd > dF;
    vector< MatrixXd > dPinf;
    vector< double > dR;
    for (int j=0; j<nparam; j++) {
        
        MexMat foo1 (mxGetPr(prhs[6])+dim*dim*j,dim,dim);
        dF.push_back(foo1);

        MexMat foo2 (mxGetPr(prhs[7])+dim*dim*j,dim,dim);
        dPinf.push_back(foo2);
        
        MexMat foo3 (mxGetPr(prhs[8])+j,1,1);
        dR.push_back(foo3(0));
        
    }

    // Set up the infinite-horizon GP
    InfiniteHorizonGP gp(dt,F,H,Pinf,R,dF,dPinf,dR);

    // Loop through data
    for (long int k = 0; k < y.rows(); k++) {
        gp.update(y(k));
    }

    // Left hand side (plhs)
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,nparam,mxREAL);
    *mxGetPr(plhs[0]) = gp.getLik();
    MexVec out(mxGetPr(plhs[1]),nparam);
    out = gp.getLikDeriv();
    
}

