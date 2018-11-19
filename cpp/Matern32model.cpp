#include <vector>
#include <cmath>
#include <Eigen/Core>

#include "Matern32model.hpp"

Matern32model::Matern32model()
{
    magnSigma2 = 1.0;
    lengthScale = 1.0;
    sigma2 = 1.0;
    Matern32model::updateModel();
}

void Matern32model::updateModel()
{
    // Model matrices for Matern v=3/2
    double lambda = sqrt(3.0)/lengthScale;
    F.setZero(2,2); Pinf.setZero(2,2); H.setZero(1,2);
    F << 0.0, 1.0, -lambda*lambda, -2*lambda;
    Pinf << magnSigma2, 0.0, 0.0, magnSigma2*lambda*lambda;
    H << 1.0, 0.0;
    R = sigma2;
    
    // Derivatives dF
    dF.clear();
    dF.push_back(Eigen::MatrixXd::Zero(2,2));
    dF.push_back(Eigen::MatrixXd::Zero(2,2));
    Eigen::MatrixXd foo(2,2);
    foo << 0, 0, 6/lengthScale/lengthScale/lengthScale, 2*lambda/lengthScale;
    dF.push_back(foo);
    
    // Derivatives dPinf
    dPinf.clear();
    dPinf.push_back(Eigen::MatrixXd::Zero(2,2));
    foo << 1.0, 0, 0, 3.0/lengthScale/lengthScale;
    dPinf.push_back(foo);
    foo << 0, 0, 0,-6*magnSigma2/lengthScale/lengthScale/lengthScale;
    dPinf.push_back(foo);
    
    // Derivatives dR
    dR.clear();
    dR.push_back(1.0);
    dR.push_back(0.0);
    dR.push_back(0.0);
}

void Matern32model::setMagnSigma2(const double &val)
{
    magnSigma2 = val;
    Matern32model::updateModel();
}

void Matern32model::setLengthScale(const double &val)
{
    lengthScale = val;
    Matern32model::updateModel();
}

void Matern32model::setSigma2(const double &val)
{
    sigma2 = val;
    Matern32model::updateModel();
}

MatrixXd Matern32model::getF()
{
    return F;
}

MatrixXd Matern32model::getPinf()
{
    return Pinf;
}

MatrixXd Matern32model::getH()
{
    return H;
}

double Matern32model::getR()
{
    return R;
}

vector< MatrixXd > Matern32model::getdF()
{
   return dF;
}

vector< MatrixXd > Matern32model::getdPinf()
{
    return dPinf;
}

vector< double > Matern32model::getdR()
{
    return dR;
}

double Matern32model::getMagnSigma2()
{
    return magnSigma2;
}

double Matern32model::getLengthScale()
{
    return lengthScale;
}

double Matern32model::getSigma2()
{
    return sigma2;
}


