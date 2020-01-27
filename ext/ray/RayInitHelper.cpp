#include "ray_src/FF.h"
#include "mex.hpp"
#include "mexAdapter.hpp"

namespace md = matlab::data;

class MexFunction : public matlab::mex::Function {
    md::ArrayFactory factory;
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        md::TypedArray<double> normals = std::move(inputs[0]);
        md::TypedArray<int> tets = std::move(inputs[1]);
        int nv = normals.getDimensions()[1];
        int nt = tets.getDimensions()[1];

        md::buffer_ptr_t<double> rot = factory.createBuffer<double>(9 * nv);

        computeFF(nv, nt, normals.release().get(), tets.release().get(), rot.get());
        
        outputs[0] = factory.createArrayFromBuffer({9, static_cast<size_t>(nv)}, std::move(rot));
    }
};