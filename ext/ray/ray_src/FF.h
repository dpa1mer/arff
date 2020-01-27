
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>
#include <iostream>


//   ___________                  _
//  |____ |  _  \                | |
//      / / | | | __   _____  ___| |_ ___  _ __
//      \ \ | | | \ \ / / _ \/ __| __/ _ \| '__|
//  .___/ / |/ /   \ V /  __/ (__| || (_) | |
//  \____/|___/     \_/ \___|\___|\__\___/|_|

// nothing special here
struct vec3 {
    vec3(double* ptr)                                    { data[0] = ptr[0]; data[1] = ptr[1]; data[2] = ptr[2]; }
    vec3(double p_x = 0, double p_y = 0, double p_z = 0) { data[0] = p_x; data[1] = p_y; data[2] = p_z; }
    vec3 normalized()                                    { double n = norm(); return vec3(data[0] / n, data[1] / n, data[2] / n); }
    double& operator[](int i)                            { return data[i]; }
    vec3 operator*(double & s)                           { return vec3(s*data[0], s*data[1], s*data[2]); }
    double norm2()                                       { return data[0] * data[0] + data[1] * data[1] + data[2] * data[2]; }
    double norm()                                        { return sqrt(norm2()); }
    double data[3];
};
inline double operator * (vec3& a, vec3& b)              { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];}
inline vec3 operator + (vec3 a, vec3 b)                  { return vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);}
inline vec3 operator * (double s, vec3& a)               { return vec3(s*a[0], s*a[1], s*a[2]);}
inline vec3 cross(vec3 a, vec3 b)                        { return vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);}

//   _____             _                  _
//  |  _  |           | |                (_)
//  | | | |_   _  __ _| |_ ___ _ __ _ __  _  ___  _ __
//  | | | | | | |/ _` | __/ _ \ '__| '_ \| |/ _ \| '_ \
//  \ \/' / |_| | (_| | ||  __/ |  | | | | | (_) | | | |
//   \_/\_\\__,_|\__,_|\__\___|_|  |_| |_|_|\___/|_| |_|



struct Quaternion {

    // BEWARE :  takes the axis of rotation and the angle of rotation
    Quaternion(vec3 axis= vec3(0, 0, 0), double alpha = 0) {
        v = sin(alpha / 2.0)*axis;
        s = cos(alpha / 2.0);
    }
    double norm2() { return v.norm2() + s*s; }

    // convert into a 3x3 rotation matrix
    void to_matrix(double * mat) {
        double t, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;
        t = 2.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + s * s);

        xs = v[0] * t;
        ys = v[1] * t;
        zs = v[2] * t;

        wx = s * xs;
        wy = s * ys;
        wz = s * zs;

        xx = v[0] * xs;
        xy = v[0] * ys;
        xz = v[0] * zs;

        yy = v[1] * ys;
        yz = v[1] * zs;
        zz = v[2] * zs;

        mat[0] = 1.0 - (yy + zz);
        mat[1] = xy + wz;
        mat[2] = xz - wy;
        mat[3] = xy - wz;
        mat[4] = 1.0 - (xx + zz);
        mat[5] = yz + wx;
        mat[6] = xz + wy;
        mat[7] = yz - wx;
        mat[8] = 1.0 - (xx + yy);
    }

    // convert into Euler's angles
    void to_Euler(double &psi, double &theta, double &phi) {
        double R[9];
        to_matrix(R);
        if (fabs(fabs(R[6]) - 1.) < 1e-5) {
            phi = 0;
            double tmp = atan2(R[1], R[2]);
            theta = -R[6] * M_PI / 2.;
            psi = -R[6] * phi + tmp;
        }
        else {
            theta = -asin(R[6]);
            double c = cos(theta);
            psi = atan2(R[7] / c, R[8] / c); // TODO strictly speaking we do not need to divide by c here
            phi = atan2(R[3] / c, R[0] / c); // careful sign choice would be sufficient
        }
    }
    vec3 v;
    double s;
};

inline Quaternion operator * (Quaternion a, Quaternion b) {
    Quaternion res;
    res.s = a.s * b.s - a.v * b.v;
    res.v = a.s * b.v + b.s * a.v + cross(a.v, b.v);
    return res;
}





//   _____       _               _           _   _   _                                  _
//  /  ___|     | |             (_)         | | | | | |                                (_)
//  \ `--. _ __ | |__   ___ _ __ _  ___ __ _| | | |_| | __ _ _ __ _ __ ___   ___  _ __  _  ___ ___
//   `--. \ '_ \| '_ \ / _ \ '__| |/ __/ _` | | |  _  |/ _` | '__| '_ ` _ \ / _ \| '_ \| |/ __/ __|
//  /\__/ / |_) | | | |  __/ |  | | (_| (_| | | | | | | (_| | |  | | | | | | (_) | | | | | (__\__ \
//  \____/| .__/|_| |_|\___|_|  |_|\___\__,_|_| \_| |_/\__,_|_|  |_| |_| |_|\___/|_| |_|_|\___|___/
//        | |
//        |_|

struct  SphericalHarmonicL4 {
    double coeff[9];

    SphericalHarmonicL4(double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8) {
        coeff[0] = x0; coeff[1] = x1; coeff[2] = x2; coeff[3] = x3; coeff[4] = x4; coeff[5] = x5; coeff[6] = x6; coeff[7] = x7; coeff[8] = x8;
    }

    SphericalHarmonicL4() {
        for (int i = 9; i--; coeff[i] = 0.);
    }

    double& operator[](int i) {
        return coeff[i];
    }

    void mult9(double M[9][9]) { // M*coeff, coeff being column vector
        SphericalHarmonicL4 copy(*this);
        for (int row=0; row<9; row++) {
            coeff[row] = 0;
            for (int column=0; column<9; column++) {
                coeff[row] += M[row][column]*copy[column];
            }
        }
    }

    void Rz(double a) {
        double M[9][9] = {{cos(4.*a),    0,             0,             0,             0,             0,             0,             0,             sin(4.*a) },
                          {0,            cos(3.*a),     0,             0,             0,             0,             0,             sin(3.*a),     0         },
                          {0,            0,             cos(2.*a),     0,             0,             0,             sin(2.*a),     0,             0         },
                          {0,            0,             0,             cos(a),        0,             sin(a),        0,             0,             0         },
                          {0,            0,             0,             0,             1,             0,             0,             0,             0         },
                          {0,            0,             0,             -sin(a),       0,             cos(a),        0,             0,             0         },
                          {0,            0,             -sin(2.*a),    0,             0,             0,             cos(2.*a),     0,             0         },
                          {0,            -sin(3.*a),    0,             0,             0,             0,             0,             cos(3.*a),     0         },
                          {-sin(4.*a),   0,             0,             0,             0,             0,             0,             0,             cos(4.*a) }};
        mult9(M);
    }

    void Rx90() {
        // just in case, inverse of the matrix is equal to its transpose
        double M[9][9] = {{0,                  0,                0,                  0,                0,                 std::sqrt(14.)/4., 0,                 -std::sqrt(2.)/4., 0                },
                          {0,                  -3./4.,           0,                  std::sqrt(7.)/4., 0,                 0,                 0,                 0,                 0                },
                          {0,                  0,                0,                  0,                0,                 std::sqrt(2.)/4.,  0,                 std::sqrt(14.)/4., 0                },
                          {0,                  std::sqrt(7.)/4., 0,                  3./4.,            0,                 0,                 0,                 0,                 0                },
                          {0,                  0,                0,                  0,                3./8.,             0,                 std::sqrt(5.)/4.,  0,                 std::sqrt(35.)/8.},
                          {-std::sqrt(14.)/4., 0,                -std::sqrt(2.)/4.,  0,                0,                 0,                 0,                 0,                 0                },
                          {0,                  0,                0,                  0,                std::sqrt(5.)/4.,  0,                 1./2.,             0,                 -std::sqrt(7.)/4.},
                          {std::sqrt(2.)/4.,   0,                -std::sqrt(14.)/4., 0,                0,                 0,                 0,                 0,                 0                },
                          {0,                  0,                0,                  0,                std::sqrt(35.)/8., 0,                 -std::sqrt(7.)/4., 0,                 1./8.            }};
        mult9(M);
    }

    void RxMinus90() {
        // just in case, inverse of the matrix is equal to its transpose
        double M[9][9] = {{0,                 0,                0,                 0,                0,                 -std::sqrt(14.)/4., 0,                 std::sqrt(2.)/4.,   0                },
                          {0,                 -3./4.,           0,                 std::sqrt(7.)/4., 0,                 0,                  0,                 0,                  0                },
                          {0,                 0,                0,                 0,                0,                 -std::sqrt(2.)/4.,  0,                 -std::sqrt(14.)/4., 0                },
                          {0,                 std::sqrt(7.)/4., 0,                 3./4.,            0,                 0,                  0,                 0,                  0                },
                          {0,                 0,                0,                 0,                3./8.,             0,                  std::sqrt(5.)/4.,  0,                  std::sqrt(35.)/8.},
                          {std::sqrt(14.)/4., 0,                std::sqrt(2.)/4.,  0,                0,                 0,                  0,                 0,                  0                },
                          {0,                 0,                0,                 0,                std::sqrt(5.)/4.,  0,                  1./2.,             0,                  -std::sqrt(7.)/4.},
                          {-std::sqrt(2.)/4., 0,                std::sqrt(14.)/4., 0,                0,                 0,                  0,                 0,                  0                },
                          {0,                 0,                0,                 0,                std::sqrt(35.)/8., 0,                  -std::sqrt(7.)/4., 0,                  1./8.            }};
        mult9(M);
    }

    void Ry(double alpha) {
        Rx90();
        Rz(alpha);
        RxMinus90();
    }

    void Rx(double alpha) {
        Ry(-M_PI/2);
        Rz(alpha);
        Ry(M_PI/2);
    }

    void Rot(Quaternion rv) {
        double alpha, beta, gamma;
        rv.to_Euler(alpha, beta, gamma);
        Rx(alpha);
        Ry(beta);
        Rz(gamma);
    }

    SphericalHarmonicL4 Ex() {
        return SphericalHarmonicL4(-sqrt(2.)*coeff[7], -sqrt(2.)*coeff[8] - sqrt(3.5)*coeff[6], -sqrt(3.5)*coeff[7] - sqrt(4.5)*coeff[5], -sqrt(4.5)*coeff[6] - sqrt(10.)*coeff[4], sqrt(10.)*coeff[3], sqrt(4.5)*coeff[2], sqrt(3.5)*coeff[1] + sqrt(4.5)*coeff[3], sqrt(2.)*coeff[0] + sqrt(3.5)*coeff[2], sqrt(2.)*coeff[1]);
    }

    SphericalHarmonicL4 Ey() {
        return SphericalHarmonicL4(sqrt(2.)*coeff[1], -sqrt(2.)*coeff[0] + sqrt(3.5)*coeff[2], -sqrt(3.5)*coeff[1] + sqrt(4.5)*coeff[3], -sqrt(4.5)*coeff[2], -sqrt(10.)*coeff[5], -sqrt(4.5)*coeff[6] + sqrt(10.)*coeff[4], -sqrt(3.5)*coeff[7] + sqrt(4.5)*coeff[5], -sqrt(2.)*coeff[8] + sqrt(3.5)*coeff[6], sqrt(2.)*coeff[7]);
    }

    SphericalHarmonicL4 Ez() {
        return SphericalHarmonicL4(4 * coeff[8], 3 * coeff[7], 2 * coeff[6], coeff[5], 0, -coeff[3], -2 * coeff[2], -3 * coeff[1], -4 * coeff[0]);
    }

    double operator *(const SphericalHarmonicL4 &other) {
        double res = 0;
        for (int d = 0; d < 9; d++) res += coeff[d] * other.coeff[d];
        return res;
    }

    double norm() {
        return sqrt((*this)*(*this));
    }

    void normalize() {
        double s = norm();
        for (int d = 0; d<9; d++) coeff[d] /= s;
    }

    static
    std::pair<Quaternion, SphericalHarmonicL4>
    project_helper(SphericalHarmonicL4 query, double grad_threshold, double dot_threshold) {
        SphericalHarmonicL4 init_harmonics[5] = {
            SphericalHarmonicL4(0, 0, 0, 0, std::sqrt(7. / 12.), 0, 0, 0, std::sqrt(5. / 12.)),
            SphericalHarmonicL4(0, 0, 0, 0, -0.190941, 0, -0.853913, 0, 0.484123),
            SphericalHarmonicL4(0, 0, 0, 0, -0.190941, 0, 0.853913, 0, 0.484123),
            SphericalHarmonicL4(0, 0, 0, 0, 0.763763, 0, 0, 0, -0.645497),
            SphericalHarmonicL4(0, 0, -0.853913, 0, -0.190941, 0, 0, 0, -0.484123)
        };
        Quaternion init_rot[5] = { Quaternion(vec3(1,0,0),0), Quaternion(vec3(1, 0, 0),0.78539816339), Quaternion(vec3(0, 1, 0),0.78539816339), Quaternion(vec3(0, 0, 1),0.78539816339), Quaternion(vec3(0.70710678118,0.70710678118,0),0.78539816339) };
        Quaternion W;
        SphericalHarmonicL4 v;
        double dot = -1.;
        query.normalize();

        for (int i = 0; i < 5; i++) {
            double tdot = init_harmonics[i] * query;
            if (tdot > dot) {
                dot = tdot;
                W = init_rot[i];
                v = init_harmonics[i];
            }
        }
        
//         W = init_rot[0];
//         v = init_harmonics[0];
//         dot = init_harmonics[0] * query;

        int cnt = 0;
        double olddot = dot;
        while (cnt < 10000) {
            vec3 grad(query*v.Ex(), query*v.Ey(), query*v.Ez());
            if (grad.norm() < grad_threshold) break;
            grad = (1. / 8.)*grad; // empirically found constant
            v.Rx(grad[0]); v.Ry(grad[1]); v.Rz(grad[2]);
            W = Quaternion(vec3(1, 0, 0), grad[0])*W;
            W = Quaternion(vec3(0, 1, 0), grad[1])*W;
            W = Quaternion(vec3(0, 0, 1), grad[2])*W;
            cnt++;
            dot = v*query;
            if (dot - olddot < dot_threshold) break;
            olddot = dot;
        }
        if (cnt == 10000) std::cerr << "[error] SH projection infinite loop protection" << std::endl;
        return std::make_pair(W, v);
    }


    static Quaternion project(SphericalHarmonicL4 query, double grad_threshold, double dot_threshold) {
        return project_helper(query, grad_threshold, dot_threshold).first;
    }
    
    static SphericalHarmonicL4 project_sph(SphericalHarmonicL4 query, double grad_threshold, double dot_threshold) {
        return project_helper(query, grad_threshold, dot_threshold).second;
    }
};



void computeFF(
	int nbv,		// number of vertices
	int nbt,		// number of tetrahedrons
	double *n1,		// INPUT: n1[3*v+d] is the d^{th} coordinate of the first locked normal of vertex v
	int	*tet,		// INPUT: tet[4*t+] is the i^{th} vertex d of tet t
	double	*rot	// OUTPUT: rot[9*v+3*e+d] is the d^{th} coordinate of the e^{th} vector of vertex v
);
