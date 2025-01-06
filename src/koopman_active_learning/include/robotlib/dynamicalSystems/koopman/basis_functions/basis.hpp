#ifndef RIGID_BODY_BASIS_HPP
#define RIGID_BODY_BASIS_HPP
#include <armadillo>
#include <math.h>
#include "basis_template.hpp"


class RigidBodyBasis : public Basis {
    int NU;
    arma::mat dphidx;
    arma::mat dphidu;
public:
    RigidBodyBasis(int NX = 14,  int NK = 49, int ORDER=1 ) : Basis(NX, NK, ORDER) {
        NU = 7;
        dphidx = arma::zeros<arma::mat>(NK, NX);
        dphidu = arma::zeros<arma::mat>(NK, NU);
    }
    arma::vec fk(const arma::vec& x, const arma::vec& u) {

        arma::vec basisVec = arma::join_cols(x, u);
        basisVec = arma::join_cols(basisVec, arma::pow(x,3));        
        basisVec = arma::join_cols(basisVec, x.head_rows(13) % x.tail_rows(13));

        basisVec = arma::join_cols(basisVec, arma::vec({1}));

        return basisVec;
    }

    arma::vec zk(const arma::vec& x) {
        arma::vec basisVec = arma::join_cols(x, arma::pow(x,3));
        basisVec = arma::join_cols(basisVec, x.head_rows(13) % x.tail_rows(13));
        basisVec = arma::join_cols(basisVec, arma::vec({1}));
    
        return basisVec;
    }

    arma::mat dfkdx(const arma::vec& x, const arma::vec& u) {
        dphidx.zeros();
        dphidx.submat(0,0, NX-1, NX-1) = arma::eye<arma::mat>(NX, NX);
        dphidx.submat(NX+NU, 0, 2*NX+NU-1, NX-1 ) = arma::diagmat(3*arma::pow(x,2));

        dphidx.submat(2*NX+NU, 0, 3*NX+NU-2, NX-2) = arma::diagmat(x.tail_rows(13));
        dphidx.submat(2*NX+NU, 0, 3*NX+NU-1, NX-1) += arma::diagmat(x.head_rows(13), 1);

        return dphidx;
    }

    arma::mat dfkdu(const arma::vec& x, const arma::vec& u) {
        dphidu.submat(NX,0, NX+NU-1,NU-1) = arma::eye<arma::mat>(NU,NU); 
        return dphidu;
    }

};

#endif
