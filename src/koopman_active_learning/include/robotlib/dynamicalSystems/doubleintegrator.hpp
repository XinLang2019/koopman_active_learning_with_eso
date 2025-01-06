#ifndef DOUBLEINTEGRATOR_HPP
#define DOUBLEINTEGRATOR_HPP

#include <armadillo>
#include "system.hpp"

class DoubleIntegrator : public System
{
private:

    arma::mat A;
    arma::mat B;

    arma::vec lowpass_u = {0,0};

public:

    float dt;

    DoubleIntegrator(float _dt, int _nX=4, int _nU=2) : System(_nX, _nU){ // constructor
        dt = _dt;
        A = { {0 , 0 , dt, 0 },
              {0 , 0 , 0, dt},
              {0, 0, -0.2*dt, 0},
              {0, 0, 0, -0.2*dt},
          };
        A = A + arma::eye<arma::mat>(nX,nX);

        B = {
                {0,0},
                {0,0},
                {dt,0},
                {0,dt}
            };

    }

    arma::vec f(const arma::vec& xp,const arma::vec& u) {
        arma::vec x = A*xp + B*u;
        lowpass_u = lowpass_u - 0.2 * (lowpass_u - u);
        if (x[0] > 0.7) {
            x[2] = -1;
        } else if (x[0] < -0.7) {
            x[2] = 1;
        }

        if (x[1] > 0.4) {
            x[3] = -1;
        } else if (x[1] < -0.4) {
            x[3] = 1;
        }
        return x;

    }

    arma::mat fdx(const arma::vec& x,const arma::vec& u){
        return A;
    }

    arma::mat fdu(const arma::vec& x,const arma::vec& u){
        return B;
    }
};

#endif
