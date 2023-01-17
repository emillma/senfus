// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

namespace sym {

/**
 * This function was autogenerated. Do not modify by hand.
 *
 * Args:
 *     inputs: list
 *
 * Outputs:
 *     c: list
 */
template <typename Scalar>
void Myfunction(const std::array<Scalar, 4>& inputs, std::array<Scalar, 3>* const c = nullptr) {
  // Total ops: 3

  // Input arrays

  // Intermediate terms (0)

  // Output terms (1)
  if (c != nullptr) {
    std::array<Scalar, 3>& _c = (*c);

    _c[0] = inputs[0] + inputs[1];
    _c[1] = inputs[0] * inputs[1];
    _c[2] = inputs[0] - inputs[1];
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
