#ifndef MAT32_H_
#define MAT32_H_

using namespace std;
using namespace Eigen;

class Matern32model {

public:
    
    /* Hyperparameters */
    double magnSigma2;
    double lengthScale;
    double sigma2;
    
    /* Model matrices */
    MatrixXd F;
    MatrixXd Pinf;
    MatrixXd H;
    double R;
    
    /* Derivatives */
    vector< MatrixXd > dF;
    vector< MatrixXd > dPinf;
    vector< double > dR;
    
    /* Constructor */
    Matern32model();
    
    /* Update hyperparameters */
    void setMagnSigma2(const double &val);
    void setLengthScale(const double &val);
    void setSigma2(const double &val);
    
    /* Get model matrices */
    MatrixXd getF();
    MatrixXd getPinf();
    MatrixXd getH();
    double getR();
    vector< MatrixXd > getdF();
    vector< MatrixXd > getdPinf();
    vector< double > getdR();
    
    /* Get hyperparameters */
    double getMagnSigma2();
    double getLengthScale();
    double getSigma2();
    
private:
    
    void updateModel();
    
};

#endif
