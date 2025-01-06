#ifndef TRAJECTORY_OBJECTIVE_HPP
#define TRAJECTORY_OBJECTIVE_HPP

#include <math.h>
#include <armadillo>
#include "wrap2Pi.hpp"
#include "../dynamicalSystems/koopman/basis_functions/basis_template.hpp"
#include "barrier.hpp"


class TRAObjective {

public:
    arma::mat Q;
    arma::mat Qf;
    arma::mat R;
    arma::mat xd;
    double q_fisher = 20;//50; // 200
    Basis* basis;

    arma::mat Qfk;

    TRAObjective(arma::mat _Q, arma::mat _R, arma::mat _Qf, arma::mat _xd, Basis* _basis) {
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
    inline double l(const arma::vec& x, const arma::vec& u, int k) {
        arma::vec xn = x;
        arma::vec fk = basis->fk(xn, u);
        if (arma::as_scalar(fk.t() * fk) < 1e-10) {
            return arma::as_scalar(0.5 * (xn.t() - xd.col(k).t()) * Q * (xn - xd.col(k)));
        } else {
            return q_fisher*pow(arma::as_scalar(fk.t() * fk) ,-1) + arma::as_scalar(0.5 * (xn.t() - xd.col(k).t()) * Q * (xn - xd.col(k)));// + barrier::barr(xn);
        }
    }

    arma::vec ldx(const arma::vec& x, const arma::vec& u, int k) {
        arma::vec xn = x;
        arma::vec fk = basis->fk(xn, u);
        arma::mat dfk = basis->dfkdx(xn, u);
        if (arma::as_scalar(fk.t() * fk) < 1e-10 || dfk.has_nan()) {
            return Q * (xn - xd.col(k));
        } else {
            return Q * (xn - xd.col(k)) - q_fisher*2 * pow(arma::as_scalar(fk.t() * fk), -2) * dfk.t() * fk;// + barrier::dbarr(xn);
        }
    }

    double get_cost(const arma::mat& x, const arma::mat& u, int kstart) {
        double J = 0.0;
        int k;
        for (k = 0; k  < u.n_cols; k++ ) {
            J += l(x.col(k), u.col(k), kstart+k);
        }
        return J + arma::as_scalar((x.tail_cols(1).t() - xd.col(k).t()) * Qf * (x.tail_cols(1) - xd.col(k)) );
    }

};

#endif
