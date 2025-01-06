#ifndef KOOPMANSYSTEM_HPP
#define KOOPMANSYSTEM_HPP

#include <armadillo>
#include "../system.hpp"
#include "basis_functions/basis_template.hpp"
#include <math.h>


class KoopmanSystem : public System
{

private:
    arma::mat __A;
    arma::mat __G;
    arma::mat __K;
    arma::mat __Ktilde;
    double counter = 0;


public:

    Basis* basis;

    KoopmanSystem(Basis* _basis, int _nX=14, int _nU=7) : System(_nX, _nU) {
        basis = _basis;
        __A = arma::zeros<arma::mat>(basis->NK, basis->NK);
        __G = arma::zeros<arma::mat>(basis->NK, basis->NK);
        __K = arma::zeros<arma::mat>(basis->NK, basis->NK);
        __Ktilde = arma::ones<arma::mat>(_nX, basis->NK);
        __Ktilde = 0.001 * __Ktilde;
    }

    inline arma::vec f(const arma::vec& x, const arma::vec& u) {
        return __Ktilde * basis->fk(x, u);
    }

    inline arma::mat fdx(const arma::vec& x, const arma::vec& u) {
        return __Ktilde * basis->dfkdx(x, u);
    }
    inline arma::mat fdu(const arma::vec& x, const arma::vec& u) {
        return __Ktilde * basis->dfkdu(x, u);
    }

    inline arma::mat get_K() {
        return __K;
    }


    void gradStep(const arma::vec& dataIn, const arma::vec& cdataIn, const arma::vec& dataOut) {

        arma::vec phix = basis->fk(dataIn, cdataIn);
        arma::vec phixpo = basis->fk(dataOut, cdataIn);

        __G += phix * phix.t();
        __A += phix * phixpo.t();
       
        try {
            __K = arma::pinv(__G) * __A;
        } catch (std::runtime_error& e) {
            std::cout << "CAUGHT THAT ERROR" << std::endl;
        }
        
        // std::cout << "Koopman Cost: \t" << pCostTest.t() * pCostTest << std::endl;
        arma::mat k_temp = __K.t();
        __Ktilde = k_temp.rows(0, nX-1);
        
    }

    void gradStep_continuous(const arma::vec& dataIn, const arma::vec& cdataIn, const arma::vec& dataOut) {

        arma::vec phix = basis->fk(dataIn, cdataIn);
        arma::vec phixpo = basis->fk(dataOut, cdataIn);
        counter += 1;
        __G += (phix * phix.t() - __G)/counter;
        __A += (phix * phixpo.t() - __A)/counter;
       
        try {
            __K = arma::pinv(__G) * __A;
        } catch (std::runtime_error& e) {
            std::cout << "CAUGHT THAT ERROR" << std::endl;
        }
        arma::mat k_cont = real(arma::logmat(__K))/0.01;
        arma::mat k_temp = __K.t();
        __Ktilde = k_temp.rows(0, nX-1);
        
    }

    void saveOperator(std::string filePath) {
        __K.save(filePath + "koopman_operator.csv", arma::raw_ascii);
        __A.save(filePath + "A_operator.csv", arma::raw_ascii);
        __G.save(filePath + "G_operator.csv", arma::raw_ascii);
    }

    void loadOperator(const arma::mat& a, arma::mat& g) {
        __A = a;
        __G = g;
    }

    arma::vec get_koopman(){
        return arma::vectorise(__K);
    }

    arma::mat Inv_koopman(){
        arma::mat k = __Ktilde.cols(__Ktilde.n_cols - 7, __Ktilde.n_cols - 1);
        
        return arma::pinv(k);
    }

    void showOperator() {
        std::cout << __Ktilde.row(0) << std::endl;
    }

    void ResetKoopmanMatrices(Basis* newBasis) {
        arma::mat __Atemp = __A;
        arma::mat __Gtemp = __G;
        arma::mat __Ktemp = __K;
        arma::mat __Ktildetemp = __Ktilde;
        __A = arma::zeros<arma::mat>(newBasis->NK, newBasis->NK);
        __G = arma::zeros<arma::mat>(newBasis->NK, newBasis->NK);
        __K = arma::zeros<arma::mat>(newBasis->NK, newBasis->NK);
        __Ktilde = arma::zeros<arma::mat>(nX, newBasis->NK);

        __A.submat(0,0,__Atemp.n_rows-1, __Atemp.n_cols-1) = __Atemp;
        __G.submat(0,0,__Gtemp.n_rows-1, __Gtemp.n_cols-1) = __Gtemp;
        __K.submat(0,0,__Ktemp.n_rows-1, __Ktemp.n_cols-1) = __Ktemp;
        __Ktilde.submat(0,0,__Ktildetemp.n_rows-1, __Ktildetemp.n_cols-1) = __Ktildetemp;

        basis = newBasis; // update with new pointer
    }

};


#endif
