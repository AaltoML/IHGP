#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <unsupported/Eigen/MatrixFunctions>

#include "InfiniteHorizonGP.hpp"

const double InfiniteHorizonGP::DARE_EPS = 1e-10;
const int InfiniteHorizonGP::DARE_MAXIT = 100;

InfiniteHorizonGP::InfiniteHorizonGP(const double dt, const Eigen::MatrixXd &F, const Eigen::MatrixXd &HH, const Eigen::MatrixXd &Pinf, const double &R, const std::vector<Eigen::MatrixXd> &dF, const std::vector<Eigen::MatrixXd> &dPinf, const std::vector<double> &dR)
{
    // Solve A and Q (stationary systems)
    A = (F*dt).exp();
    Q = Pinf - A*Pinf*A.transpose();

    // Assign measurement model
    H = HH;
    
    // Solve the discrete algebraic Riccati equation
    Eigen::MatrixXd PP = InfiniteHorizonGP::DARE(A,H,Q,R);
    
    // Stationary innovation variance
    S = (H*PP*H.transpose())(0) + R;
    
    // Stationary gain
    K = PP*H.transpose()/S;
    
    // State covariance
    PF = PP - K*H*PP;
    
    // Pre-calculate
    HA = (H*A).transpose();
    AKHA = A-K*H*A;
    
    // Number of parameters
    nparam = (int)dF.size();
    
    // State dimensionality
    int dim = (int)F.rows();
    
    // Initialize mean and helpers
    m.setZero(dim);
    
    // Prepare
    Eigen::MatrixXd AK = A*K;
    Eigen::MatrixXd FF(2*dim,2*dim);
    FF.setZero();
    FF.topLeftCorner(dim,dim) = F;
    FF.bottomRightCorner(dim,dim) = F;
    
    // Allocate the arrays
    HdA  = new Eigen::VectorXd[nparam];
    dK = new Eigen::VectorXd[nparam];
    dAKHA = new Eigen::MatrixXd[nparam];
    dS = new double[nparam];
    dm = new Eigen::VectorXd[nparam];
    
    // Pre-calculate the needed derivative matrices
    for (int j=0; j<nparam; j++)
    {
        // Assign derivative of drift model matrix
        FF.bottomLeftCorner(dim,dim) = dF[j];
        
        // Solve the matrix exponential (see ...)
        Eigen::MatrixXd AA = (FF*dt).exp();
        
        // Extract the derivative of the discrete-time dynamic model
        Eigen::MatrixXd dA = AA.bottomLeftCorner(dim,dim);
        Eigen::MatrixXd dQ = dPinf[j] - dA*Pinf*A.transpose() - A*dPinf[j]*A.transpose() - A*Pinf*dA.transpose();
        dQ = .5*(dQ+dQ.transpose()).eval();
        
        // Precalculate C
        Eigen::MatrixXd C = dA*PP*A.transpose() + A*PP*dA.transpose() - dA*PP*H.transpose()*AK.transpose() - AK*H*PP*dA.transpose() + AK*dR[j]*AK.transpose() + dQ;
        C = .5*(C+C.transpose()).eval();

        // Solve DARE
        Eigen::MatrixXd dPP = InfiniteHorizonGP::DARE(A-AK*H,Eigen::MatrixXd::Zero(dim,dim),C,0.0);
        
        // Evaluate dS and dK
        dS[j] = (H*dPP*H.transpose())(0) + dR[j];
        dK[j] = dPP*H.transpose()/S - PP*H.transpose()*(((H*dPP*H.transpose())(0)+dR[j])/S/S);
        dAKHA[j] = dA - dK[j]*H*A - K*H*dA;
        HdA[j] = (H*dA).transpose();

        // Initial dm
        dm[j] = Eigen::VectorXd::Zero(dim);
        
    }

    // Initialize log likelihood and its gradient
    edata = 0;
    gdata = Eigen::VectorXd::Zero(nparam);
    
}

InfiniteHorizonGP::~InfiniteHorizonGP()
{
    delete[] HdA;
    delete[] dK;
    delete[] dAKHA;
    delete[] dS;
    delete[] dm;
}
    
void InfiniteHorizonGP::update(const double &y)
{
    // Define constants
    const double PI = 3.141592654;
    
    // Innovation mean
    double v = y - HA.dot(m);

    // Update marginal likelihood
    edata += .5*v*v/S + .5*log(2*PI) + .5*log(S);
    
    // Update derivatives
    for (int j=0; j<nparam; j++)
    {
        // Derivatives of innovation mean
        double dv = -HdA[j].dot(m) - HA.dot(dm[j]);

        // Derivatives of marginal likelihood
        gdata(j) += v*dv/S - .5*v*v*dS[j]/S/S + .5*dS[j]/S;

        // Derivatives of state mean
        dm[j] = dAKHA[j]*m + AKHA*dm[j] + dK[j]*y;
    }
    
    // Recursion for the state mean
    m = AKHA*m + K*y;

    // Store state mean for possible backward pass
    MF.push_back(m);
    
}

std::vector<double> InfiniteHorizonGP::getEft()
{
    // Solve backward smoother gain
    int dim = (int)A.rows();
    Eigen::MatrixXd PP = A*PF*A.transpose()+Q;
    Eigen::LDLT<Eigen::MatrixXd> PPldl = PP.ldlt();
    Eigen::MatrixXd G = PPldl.solve(A*PF).transpose();
    
    // Solve smoother state covariance
    Eigen::MatrixXd QQ = PF-G*PP*G.transpose();
    QQ = .5*(QQ+QQ.transpose()).eval();
    P = InfiniteHorizonGP::DARE(G,Eigen::MatrixXd::Zero(dim,dim),QQ,0.0);
    
    // Output vector
    std::vector<double> Eft;
    
    // Initialize with last element
    m = MF[MF.size()-1];
    Eft.push_back((H*m)(0));
    
    // Run backward pass
    for (int k=(int)MF.size()-2; k>=0; k--)
    {
        m = MF[k] + G*(m-A*MF[k]);
        Eft.push_back((H*m)(0));
    }
    
    // Reverse
    std::reverse(Eft.begin(),Eft.end());
    
    return Eft;
    
}

double InfiniteHorizonGP::getVarft()
{
    return (H*P*H.transpose())(0);
}

double InfiniteHorizonGP::getLik()
{
    return edata;
}

Eigen::VectorXd InfiniteHorizonGP::getLikDeriv()
{
    return gdata;
}

Eigen::MatrixXd InfiniteHorizonGP::DARE(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &Q, const double &R)
{

    // Initial guess
    int dim = (int)A.rows();
    Eigen::MatrixXd X(dim,dim);
    Eigen::MatrixXd X_prev(dim,dim);
    Eigen::MatrixXd K(dim,B.rows());
    X.setIdentity();
    
    // Number of loops
    int n = 0;
    
    // Iterate a maximum of 100 iterations
    while (n < DARE_MAXIT)
    {
        // Step
        n++;
        X_prev = X;

        // Gain (NB: Does not work in the general case, but for scalar R and possibly zero B)
        if (abs(R) < 1e-15)
        {
            K.setZero();
        } else {
            K = A*(X*B.transpose() / ((B*X*B.transpose())(0)+R));
        }
        
        // Recursion
        X = (A - K*B)*X*(A - K*B).transpose() + K*R*K.transpose() + Q;
        
        // Check if we should break (use the Frobenius norm)
        if ((X-X_prev).norm() < DARE_EPS)
        {
            break;
        }
    }
    
    return X;
}
