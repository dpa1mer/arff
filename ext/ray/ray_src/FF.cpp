
#include "OpenNL_psm.h"
#include "FF.h"


//   _____              _       _ _   _       _ _          _   _
//  |  _  |            (_)     (_) | (_)     | (_)        | | (_)
//  | | | |_   _ _ __   _ _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
//  | | | | | | | '__| | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
//  \ \_/ / |_| | |    | | | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
//   \___/ \__,_|_|    |_|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|


void computeFF(
    int nbv,        // number of vertices
    int nbt,        // number of tetrahedrons
    double *n1,        // INPUT: n1[3*v+d] is the d^{th} coordinate of the first locked normal of vertex v
    int    *tet,        // INPUT: tet[4*t+i] is the i^{th} vertex d of tet t
    double    *rot    // OUTPUT: rot[9*v+3*e+d] is the d^{th} coordinate of the e^{th} vector of vertex v
) {

    /* Create and initialize OpenNL context */
    nlNewContext();
    nlSolverParameteri(NL_NB_VARIABLES, 11 * nbv);
    nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
    nlSolverParameteri(NL_SOLVER, 0);
    nlBegin(NL_SYSTEM);
    nlBegin(NL_MATRIX);

    // smoothing term
    for (int t = 0; t < nbt; t++) for (int v0 = 0; v0 < 4; v0++) for (int v1 = v0 + 1; v1 < 4; v1++) for (int d = 0; d < 9; d++) {
        nlBegin(NL_ROW);
        nlCoefficient(11 * tet[4 * t + v0] + d, 1.0);
        nlCoefficient(11 * tet[4 * t + v1] + d, -1.0);
        nlEnd(NL_ROW);
    }

    // boundary condition enforced by penalty energy
    for (int v = 0; v < nbv; v++) {
        vec3 n(n1 + 3 * v);
        if (n.norm() == 0) continue;
        Quaternion q;
        if (std::abs(n.normalized()[2]) < .99) {
            vec3 axis = cross(n.normalized(), vec3(0, 0, 1));
            if (std::abs(axis.norm()) > 1)
                axis = .99*(1. / axis.norm())*axis;
            q = Quaternion(axis.normalized(), atan2(axis.norm(), n.normalized()[2]));
        }

        SphericalHarmonicL4 sh0, sh4, sh8;
        sh4[4] = std::sqrt(7. / 12.);
        sh0[0] = std::sqrt(5. / 12.);
        sh8[8] = std::sqrt(5. / 12.);
        sh4.Rot(q);
        sh0.Rot(q);
        sh8.Rot(q);
        for (int d = 0; d < 9; d++) {
            nlRowScaling(100.);
            nlBegin(NL_ROW);
            nlCoefficient(11 * v + d, 1.0);
            nlCoefficient(11 * v + 9, sh0[d]);
            nlCoefficient(11 * v + 10, sh8[d]);
            nlRightHandSide(sh4[d]);
            nlEnd(NL_ROW);
        }
    }
    nlEnd(NL_MATRIX);
    nlEnd(NL_SYSTEM);
    nlSolve();

    for (int v = 0; v < nbv; v++) {
        SphericalHarmonicL4 sh;
        for (int d = 0; d < 9; d++) sh[d] = nlGetVariable(v * 11 + d);
        Quaternion quat = SphericalHarmonicL4::project(sh, 1e-3, 1e-5);
        double m[9];
        quat.to_matrix(m);
        for (int e = 0; e < 3; e++) for (int d = 0; d < 3; d++)
            rot[9 * v + 3 * e + d] = m[3 * e + d];
    }
    nlDeleteContext(nlGetCurrent());
}
