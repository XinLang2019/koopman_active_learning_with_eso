#ifndef BARRIER_HPP
#define BARRIER_HPP

#include <armadillo>
#include <math.h>

namespace barrier {

    double b_lim = M_PI/2;
    double barr_weight = 100000;

double barr(arma::vec x){
    double barr_temp = 0;


    if (x(0) >= b_lim){
        barr_temp += barr_weight *
                        pow(x(0) - b_lim, 2);
    }
    else if (x(0) <= -b_lim){
        barr_temp += barr_weight  * pow(x(0) - b_lim, 2);
    }


    return barr_temp;
}

arma::vec dbarr(arma::vec x){
    arma::vec dbarr_temp = arma::zeros<arma::vec>(x.n_rows);
    if (x(0) >= b_lim){
        dbarr_temp(0) +=  2*barr_weight  * (x(0) - b_lim);
    }
    else if (x(0) <= -b_lim){
        dbarr_temp(0) += 2*barr_weight  * (x(0) - b_lim);
    }

    return dbarr_temp;
}

}
#endif
