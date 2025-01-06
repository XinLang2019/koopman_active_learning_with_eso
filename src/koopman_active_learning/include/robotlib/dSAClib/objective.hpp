#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include <math.h>
#include <armadillo>
#include "wrap2Pi.hpp"
#include "../dynamicalSystems/koopman/basis_functions/basis_template.hpp"
#include "barrier.hpp"


class Objective {

public:
    arma::mat Q;
    arma::mat Qf;
    arma::mat R;
    arma::mat xd;
    double q_fisher = 20;//50; // 200
    Basis* basis;

    arma::mat Qfk;

    Objective(arma::mat _Q, arma::mat _R, arma::mat _Qf, arma::vec _xd, Basis* _basis) {
        Q = _Q;
        Qf = _Qf;
        R = _R;
        xd = _xd;
        basis = _basis;
        arma::vec Qfkdiag = arma::ones<arma::vec>(_basis->NK);

        Qfk = arma::diagmat(Qfkdiag);

    }
    double getFisherInformation(const arma::vec& x, const arma::vec& u) {
        arma::vec xn = x;
        arma::vec fk = basis->fk(xn,u);
        return pow(arma::as_scalar(fk.t() * Qfk * fk), -1);
    }
    inline double l(const arma::vec& x, const arma::vec& u) {
        arma::vec xn = x;
        arma::vec fk = basis->fk(xn, u);
        if (arma::as_scalar(fk.t() * fk) < 1e-10) {
            return arma::as_scalar(0.5 * (xn.t() - xd.t()) * Q * (xn - xd));
        } else {
            return q_fisher*pow(arma::as_scalar(fk.t() * fk) ,-1) + arma::as_scalar(0.5 * (xn.t() - xd.t()) * Q * (xn - xd));// + barrier::barr(xn);
        }
    }

    arma::vec ldx(const arma::vec& x, const arma::vec& u) {
        arma::vec xn = x;
        arma::vec fk = basis->fk(xn, u);
        arma::mat dfk = basis->dfkdx(xn, u);
        if (arma::as_scalar(fk.t() * fk) < 1e-10 || dfk.has_nan()) {
            return Q * (xn - xd);
        } else {
            // std::cout << "Q: " << Q << std::endl;
            // std::cout << "(xn - xd): " << (xn - xd) << std::endl;
            // std::cout << "q_fisher: " << q_fisher << std::endl;
            // std::cout << "dfk: " << dfk << std::endl;
            return Q * (xn - xd) - q_fisher*2 * pow(arma::as_scalar(fk.t() * fk), -2) * dfk.t() * fk;// + barrier::dbarr(xn);
        }
    }

    double get_cost(const arma::mat& x, const arma::mat& u) {
        double J = 0.0;
        for (int k = 0; k  < u.n_cols; k++ ) {
            J += l(x.col(k), u.col(k));
        }
        return J + arma::as_scalar((x.tail_cols(1).t() - xd.t()) * Qf * (x.tail_cols(1) - xd) );
    }

};

#endif
