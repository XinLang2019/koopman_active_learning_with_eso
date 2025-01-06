#ifndef DEISAC_HPP
#define DEISAC_HPP

#include <armadillo>
#include "../dynamicalSystems/system.hpp"
#include "objective.hpp"
// #include "ergodicity/erg_utility.hpp"
#include <math.h>
#include <cstdio> 

class deiSAC {

private:

    int _N;
    arma::mat uArr;
    arma::vec rho0;
    std::vector<arma::vec> xhist;
    std::vector<arma::vec> xpred;;
    arma::vec unom;

    double Jrunning = 0;

public:
    arma::vec umax;
    Objective* obj;

    System* sys;


    deiSAC(System* _sys, Objective* _obj, int N, arma::vec _umax, arma::vec _unom){
        sys = _sys;
        obj = _obj;
        _N = N;
        umax = _umax;
        unom = _unom;
        uArr = arma::zeros<arma::mat>(sys->nU, _N);
        for (int i = 0; i < _N; i++) {
            uArr.col(i) = unom;
        }
        rho0 = arma::zeros<arma::vec>(sys->nX);
    }

    ~deiSAC() {
        std::cout << "Deleting controller" << std::endl;
        delete sys;
        delete obj;
    }


    arma::mat forward_sim(const arma::vec&, const arma::mat& u);
    arma::mat back_sim(const arma::mat&);
    inline arma::vec rhok(const arma::vec&, const arma::vec&, const arma::vec&);
    arma::mat calc_ustar(const arma::mat&, const arma::mat&, const arma::mat&, double);
    arma::vec calc_dJdlam(const arma::mat&, const arma::mat&, const arma::mat&, const arma::mat&);
    arma::mat ustar_cont_search(const arma::vec& x0, const arma::mat& ustar, double Jinit) {
        int k = 1;
        double dJmin = 0;
        bool flag = false;
        double __scale = 0.2;

        arma::mat utemp = uArr + pow(__scale, k) * ustar; // copy over the ustar

        for (int i = 0; i < utemp.n_cols; i++){
            utemp.col(i) = saturate(utemp.col(i));
        }
        
        arma::mat xsol = forward_sim(x0, utemp); // forward simulate it
        // xpred.insert(xpred.begin(), xhist.begin(), xhist.end());
        double Jnew = obj->get_cost(xsol, utemp);
        while (Jnew - Jinit > dJmin) {
            k++;
            utemp = uArr + pow(__scale, k) * ustar;
            for (int i = 0; i < utemp.n_cols; i++){
                utemp.col(i) = saturate(utemp.col(i));
            }
            xsol = forward_sim(x0, utemp);
            Jnew = obj->get_cost(xsol, utemp);
            if (k > 15/*uspan.n_cols/4*/) {
                flag = true;
                break;
            }
        }
        if (flag == true) {
            // std::cout << Jnew - Jinit << std::endl;
            return uArr;
        }
        // std::cout << "condition: "<< Jnew - Jinit << std::endl;
        return utemp;
    }
    arma::vec saturate(arma::vec u){
        arma::vec u_unit = arma::normalise(u);
        arma::vec uabs = arma::abs(u);
        arma::vec usign = arma::sign(u);
        
        for (int i = 0; i < u.n_rows; i++){
            if (uabs(i) > umax(i)){
                // u(i) = usign(i)*umax(i);
                u = u_unit % umax;
                break;
            }
        }
        return u;
    }
    void shift_control_sequence(){
        arma::mat utemp = uArr.cols(1,_N-1);
        uArr.cols(0,_N-2) = utemp;
        uArr.col(_N-1) = unom;
    }
    arma::vec get_control(const arma::vec& x0, double _Jprev = 0) {

        Jrunning = _Jprev;

        arma::mat xsol, rhosol;
        arma::vec u_now, u2;
        int tau;
        double Jinit;
        xsol = forward_sim(x0, uArr);
        
        rhosol = back_sim(xsol);
        // std::cout << "rhosol:" << rhosol.col(0) << std::endl;
        // std::cout << "xsol:" << xsol.t() << std::endl;
        // std::cout << "uArr:" << uArr.t() << std::endl;
        
        Jinit = obj->get_cost(xsol, uArr) + Jrunning;
        double alpha_d = -555*Jinit;

        
        arma::mat ustar = calc_ustar(rhosol, xsol, uArr, alpha_d);
        
        // arma::vec dJdlam = calc_dJdlam(ustar, rhosol, xsol, uArr);
        // tau = dJdlam.index_min();
        // uArr.col(tau) = saturate(ustar.col(tau));
        // u2 = saturate(ustar.col(tau));
        // line_search(x0, u2, tau, Jinit, phik);
        // u_now = uArr.col(0);
        ustar.replace(arma::datum::nan, 0); // just in case sac returns nan

        uArr = ustar_cont_search(x0, ustar, Jinit);
        // std::cout << "x0:" << x0 << std::endl;
        // std::cout << "Jinit:" << Jinit << std::endl;
        // std::cout << "ustar:" << ustar << std::endl;
        // std::cout << "uArr:" << uArr << std::endl;
        // std::cout << "u:" << uArr.col(0) << std::endl;
        uArr.replace(arma::datum::nan, 0);
        
        u_now = uArr.col(0);
        // std::cout << "u_now:" << u_now << std::endl;
        shift_control_sequence();
        // std::cout << "uArr:" << uArr << std::endl;
        return u_now;
    }


};

arma::mat deiSAC::forward_sim(const arma::vec& x, const arma::mat& u) {
    xpred.clear();
    arma::mat X = arma::zeros<arma::mat>(sys->nX, _N);
    arma::vec x0 = x;
    for (int i = 0; i < _N; i++) {
        X.col(i) = x0;
        xpred.push_back(x0);
        x0 = sys->f(x0, u.col(i));
    }
    return X;
}

arma::mat deiSAC::back_sim(const arma::mat& x)
{
    arma::mat rho(sys->nX, _N);
    // rho.col(_N-1) = rho0
    
    rho.col(_N-1) = obj->Qf * (x.col(_N-1) - obj->xd);
    
    // rho.col(_N-1) = obj->Q * (x.col(0) - obj->xd);
    for (int k = _N-1; k >0; k--) {
        rho.col(k-1) = rhok(rho.col(k), x.col(k), uArr.col(k-1));
    }
    return rho;
}

inline arma::vec deiSAC::rhok(const arma::vec& rho, const arma::vec& xk, const arma::vec& uk)
{
    return obj->ldx(xk, uk) + sys->fdx(xk, uk).t() * rho;
}

arma::mat deiSAC::calc_ustar(const arma::mat& rho, const arma::mat& x, const arma::mat& u, double alpha_d)
{
    arma::mat ustar = arma::zeros<arma::mat>(sys->nU, _N);
    arma::mat B;
    arma::mat lam;
    for (int i = 0; i < u.n_cols; i++){
        B = sys->fdu(x.col(i), u.col(i));
        lam = B.t() * rho.col(i) * rho.col(i).t() * B;
        // ustar.col(i) = arma::inv(lam + obj->R.t()) *
        //             ( lam * u.col(i) + B.t() * rho.col(i) * alpha_d );
        // ustar.col(i) = arma::inv(lam + obj->R.t()) * (B.t()*rho.col(i)*alpha_d);// + u.col(i);
        ustar.col(i) = arma::inv(obj->R.t()) * (-B.t()*rho.col(i)); // + u.col(i);
    }

    return ustar;
}

arma::vec deiSAC::calc_dJdlam(const arma::mat& ustar, const arma::mat& rho, const arma::mat& x, const arma::mat& u)
{
    arma::vec dJdlam = arma::zeros<arma::vec>(ustar.n_cols);
    arma::vec f1, f2;
    for (int i = 0; i < u.n_cols; i++){
        f1 = sys->f(x.col(i), u.col(i));
        f2 = sys->f(x.col(i), ustar.col(i));
        dJdlam(i) = arma::as_scalar(rho.col(i).t() * (f2 - f1));
    }

    return dJdlam;
}

#endif
