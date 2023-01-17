#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/eigen.h>
#include "preintegrate.h"
#include "myfunc.h"
namespace py = pybind11;

template <typename Scalar>
void Preintegrate_binding(
    Eigen::Matrix<Scalar, 6, 1> *imu_noise, Eigen::Matrix<Scalar, 55, 1> *preint_prev, Eigen::Matrix<Scalar, 6, 1> *z_imu_est, Scalar dt, Eigen::Matrix<Scalar, 10, 1> *upsilon, Eigen::Matrix<Scalar, 9, 9> *cov)
{
    sym::Preintegrate<Scalar>(*reinterpret_cast<Eigen::Matrix<Scalar, 6, 1> *>(imu_noise), *reinterpret_cast<Eigen::Matrix<Scalar, 55, 1> *>(preint_prev), *reinterpret_cast<Eigen::Matrix<Scalar, 6, 1> *>(z_imu_est), dt, upsilon, cov);
}

template <typename Scalar>
void Myfunc_binding(
    Eigen::Matrix<Scalar, 10, 1> *inputs, Eigen::Matrix<Scalar, 9, 1> *output)
{
    sym::Myfunc<Scalar>(*reinterpret_cast<Eigen::Matrix<Scalar, 10, 1> *>(inputs), output);
}

PYBIND11_MODULE(mylib, m)
{
    m.def("preintegrate", &Preintegrate_binding<double>);
    m.def("myfunc", &Myfunc_binding<double>);
}
