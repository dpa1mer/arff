#include "ray_src/FF.h"
#include "Eigen/Dense"
#include "tbb/parallel_for.h"
#include "tbb/enumerable_thread_specific.h"
#include "mex.hpp"
#include "mexAdapter.hpp"

namespace md = matlab::data;

using Vec9X = Eigen::Matrix<double, 9, Eigen::Dynamic>;

class MexFunction : public matlab::mex::Function {
    md::ArrayFactory factory;
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        md::TypedArray<double> qIn = std::move(inputs[0]);
        md::ArrayDimensions dim_qIn = qIn.getDimensions();
        assert(dim_qIn[0] == 9);
        int n = dim_qIn[1];
        
        Vec9X q0 = Eigen::Map<Vec9X>(qIn.release().get(), 9, n);
        Vec9X q(9, n);
        
        tbb::parallel_for(tbb::blocked_range<int>(0, dim_qIn[1]), [&](const tbb::blocked_range<int>& rng) {
            
            for (int i = rng.begin(); i != rng.end(); ++i) {
                SphericalHarmonicL4 q0i(q0(0, i), q0(1, i), q0(2, i), q0(3, i), q0(4, i), q0(5, i), q0(6, i), q0(7, i), q0(8, i));
                SphericalHarmonicL4 qi = SphericalHarmonicL4::project_sph(q0i, 1e-8, 1e-8);
                for (int j = 0; j < 9; ++j) {
                    q(j, i) = qi[j];
                }
            }
        });
        
        outputs[0] = factory.createArray(dim_qIn, q.data(), &q.data()[9 * n]);
    }
};