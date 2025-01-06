#ifndef ESO_HPP
#define ESO_HPP

#include <armadillo>
#include <cmath>

class ESO {

public:
    float dt = 0.01;

    // float alpha1 = 0.1;  //good: 0.2
    // float alpha2 = 0.3;
    float delta = 0.01;  //good:0.1

    arma::vec alpha1 = arma::ones<arma::vec>(7) *0.5;  //good: 0.2
    arma::vec alpha2 = arma::ones<arma::vec>(7) *0.25;

    // dynamic distrubance good: (0.05,0.3,0.1) (200, 500, 2500)
    // dynamic distrubance good: (0.05,0.3,0.1) (200, 1200, 2500)
    // constant distrubance good: (0.2,0.3,0.1) (200, 500, 1000)
    // good :(0.5,0.25,0.01) (200, 1000, 1500)
    arma::vec beta01 = arma::ones<arma::vec>(7) * 200; //good:200,500,500
    arma::vec beta02 = arma::ones<arma::vec>(7) * 1000;
    arma::vec beta03 = arma::ones<arma::vec>(7) * 1500;

    arma::vec e = arma::zeros<arma::vec>(7);
    arma::vec z1 = arma::zeros<arma::vec>(7);
    arma::vec z2 = arma::zeros<arma::vec>(7);
    arma::vec z3 = arma::zeros<arma::vec>(7);

    arma::vec hat_z;
    arma::vec u;
    arma::vec model_x = arma::zeros<arma::vec>(14);
    arma::vec pre_x = arma::zeros<arma::vec>(14);
    arma::vec pre_u = arma::zeros<arma::vec>(7);

    ESO(const arma::vec & x, const arma::vec & _u, float _dt = 0.01) {
        dt = _dt;
        u = _u;
        estimate_hat(x);
    }

    arma::vec estimate_hat(const arma::vec & x, float _dt = 0.01){
        dt = _dt;
        for(int i =0; i<7; i++){
            sub_estimate_hat(x, i);
        }
       
        return z3;
    }

    void sub_estimate_hat(const arma::vec & x, int i) {
        e[i] = z1[i] - x[i];
        z1[i] = z1[i] + dt*(z2[i] - beta01[i]*e[i]);
        z2[i] = z2[i] + dt*(z3[i] - beta02[i]*fal(e[i], alpha1[i]) + model_x[i] + u[i]);
        z3[i] = z3[i] - dt*beta03[i]*fal(e[i], alpha2[i]);
    }

    float fal(float e, float alpha) {
        if (std::abs(e) > delta) {
            return std::pow(std::abs(e), alpha) * (smooth_sign(e));
        } else {
            return e / std::pow(delta, 1 - alpha);
        }
    } 

    double smooth_sign(double x, double seta = 0.0001) {
        return x / std::sqrt(x * x + seta);
    }

    void get_u(const arma::vec & _u){
        u = _u*10;
    }

    void get_dynamic(const arma::vec & _u){
        model_x = _u;
        std::cout<<_u<<std::endl;
    }

};

#endif