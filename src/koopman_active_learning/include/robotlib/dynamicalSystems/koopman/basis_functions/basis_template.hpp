#ifndef BASIS_TEMPLATE_HPP
#define BASIS_TEMPLATE_HPP

#include <armadillo>

class Basis {

public:

    int NX;
    int NK;
    int ORDER;
    Basis(int _NX, int _NK, int _ORDER) : NX(_NX), NK(_NK), ORDER(_ORDER) { ; }
    virtual ~Basis() { ; }
    virtual arma::vec fk(const arma::vec&, const arma::vec&) = 0;
    virtual arma::vec zk(const arma::vec&) = 0;
    virtual arma::mat dfkdx(const arma::vec&, const arma::vec&) = 0;
    virtual arma::mat dfkdu(const arma::vec&, const arma::vec&) = 0;

};

#endif
