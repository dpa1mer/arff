#include "mex.hpp"
#include "mexAdapter.hpp"
#include "fusion.h"
#include "tbb/parallel_for.h"
#include "tbb/enumerable_thread_specific.h"

using namespace matlab::data;
using namespace mosek::fusion;
using namespace monty;

class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        TypedArray<double> qIn = std::move(inputs[0]);
        ArrayDimensions dim_qIn = qIn.getDimensions();
        int d = dim_qIn[0];
        int n = dim_qIn[1];
        
        TypedArray<double> inA = std::move(inputs[1]);
        ArrayDimensions dimA  = inA.getDimensions();
        auto A = new_array_ptr<double, 2>(shape(dimA[0], dimA[1]));
        for (int i = 0; i < dimA[0]; ++i) {
            for (int j = 0; j < dimA[1]; ++j) {
                (*A)(i, j) = inA[i][j];
            }
        }
        
        TypedArray<double> inB = std::move(inputs[2]);
        ArrayDimensions dimB  = inB.getDimensions();
        auto b = new_array_ptr<double, 2>(shape(dimB[0], dimB[1]));
        for (int i = 0; i < dimB[0]; ++i) {
            for (int j = 0; j < dimB[1]; ++j) {
                (*b)(i, j) = inB[i][j];
            }
        }
        
        tbb::enumerable_thread_specific<Model::t> threadM([&]() {
            Model::t M = new Model(); //auto _M = finally([&]() { M->dispose(); });
            M->setSolverParam("intpntMultiThread", "off");
//             M->setSolverParam("intpntSolveForm", "dual");
            M->setSolverParam("intpntCoTolRelGap", "1.0e-12");

            auto Q = M->variable("Q", Domain::inPSDCone(d + 1));
            M->constraint(Expr::mul(A, Q->reshape(Set::make((d + 1) * (d + 1), 1))), Domain::equalsTo(b));
            return M;
        });
        
        bool saveQ = outputs.size() == 2;
        std::shared_ptr<ndarray<double, 2> > savedQ;
        if (saveQ) {
            savedQ.reset(new ndarray<double, 2>(shape((d + 1) * (d + 1), n), 0.0));
        }
        
        auto q = std::shared_ptr<ndarray<double, 2> >(new ndarray<double, 2>(shape(d, n), 0.0));
        tbb::parallel_for(0, n, 1, [&](int j){
            Model::t& myM = threadM.local();
            
            auto Q = myM->getVariable("Q");
            auto qi = Q->slice(Set::make(1, 0), Set::make(d + 1, 1));
            
            auto q0 = std::shared_ptr<ndarray<double, 2> >(new ndarray<double, 2>(shape(d, 1), 0.0));
            for (int i = 0; i < d; ++i) {
                (*q0)[i] = qIn[i][j];
            }
            myM->objective(ObjectiveSense::Minimize,
                    Expr::add(Expr::sum(Q->diag()),
                              Expr::mul(-2.0, Expr::dot(q0, qi))));
            
            myM->solve();
            auto qValue = *(qi->level());
            for (int i = 0; i < d; ++i) {
                (*q)(i, j) = qValue[i];
            }
            
            if (saveQ) {
                auto QValue = *(Q->level());
                for (int i = 0; i < (d + 1) * (d + 1); ++i) {
                    (*savedQ)(i, j) = QValue[i];
                }
            }
        });
        
        for (auto& M : threadM) {
            M->dispose();
        }
        
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < n; ++j) {
                qIn[i][j] = (*q)(i, j);
            }
        }
        outputs[0] = qIn;
        
        if (saveQ) {
            TypedArray<double> outQ = factory.createArray<double>(
                {static_cast<size_t>((d + 1) * (d + 1)), static_cast<size_t>(n)});
            for (int i = 0; i < (d + 1) * (d + 1); ++i) {
                for (int j = 0; j < n; ++j) {
                    outQ[i][j] = (*savedQ)(i, j);
                }
            }
            outputs[1] = outQ;
        }
    }
};