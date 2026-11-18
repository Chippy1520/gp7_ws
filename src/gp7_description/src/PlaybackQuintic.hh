#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace gp7 {

using State = std::array<double, 8>;

struct PlaybackSample {
  State q{}, v{}, a{};
};

// Pure kinematic interval evaluation: t and h are in seconds, and v/a are
// derivatives in seconds. No feedback, servo, or PID state is involved.
// Invalid inputs throw invalid_argument; unrepresentable results throw
// overflow_error. Evaluation never clamps time or extrapolates.
inline PlaybackSample interpolate(const PlaybackSample &left,
                                  const PlaybackSample &right,
                                  double h, double t) {
  if (!std::isfinite(h) || h <= 0.0 || !std::isfinite(t) || t < 0.0 || t > h)
    throw std::invalid_argument("quintic: require finite h > 0 and 0 <= t <= h");
  for (std::size_t i = 0; i < left.q.size(); ++i) {
    if (!std::isfinite(left.q[i]) || !std::isfinite(left.v[i]) ||
        !std::isfinite(left.a[i]) || !std::isfinite(right.q[i]) ||
        !std::isfinite(right.v[i]) || !std::isfinite(right.a[i]))
      throw std::invalid_argument("quintic: nonfinite endpoint state");
  }
  // Preserve the supplied knots exactly, including their derivatives.
  if (t == 0.0) return left;
  if (t == h) return right;

  const long double duration = h;
  const long double u = static_cast<long double>(t) / duration;
  PlaybackSample result;
  for (std::size_t i = 0; i < left.q.size(); ++i) {
    // Polynomial in normalized time u=t/h. Extended intermediates reduce
    // cancellation and avoid needless overflow when scaling derivatives.
    const long double c0 = left.q[i];
    const long double c1 = duration * left.v[i];
    const long double c2 = 0.5L * duration * duration * left.a[i];
    const long double r = static_cast<long double>(right.q[i]) - c0 - c1 - c2;
    const long double d = duration * right.v[i] - c1 - 2.0L * c2;
    const long double e = duration * duration * right.a[i] - 2.0L * c2;
    const long double c3 = 10.0L * r - 4.0L * d + 0.5L * e;
    const long double c4 = -15.0L * r + 7.0L * d - e;
    const long double c5 = 6.0L * r - 3.0L * d + 0.5L * e;
    result.q[i] = static_cast<double>(
        ((((c5 * u + c4) * u + c3) * u + c2) * u + c1) * u + c0);
    result.v[i] = static_cast<double>(
        ((((5.0L * c5 * u + 4.0L * c4) * u + 3.0L * c3) * u +
           2.0L * c2) * u + c1) / duration);
    result.a[i] = static_cast<double>(
        (((20.0L * c5 * u + 12.0L * c4) * u + 6.0L * c3) * u +
          2.0L * c2) / duration / duration);
    if (!std::isfinite(result.q[i]) || !std::isfinite(result.v[i]) ||
        !std::isfinite(result.a[i]))
      throw std::overflow_error("quintic: nonfinite interpolated state");
  }
  return result;
}

}  // namespace gp7
