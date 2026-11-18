// Standalone (from package root):
// g++ -std=c++17 -Wall -Wextra -Werror -pedantic test/playback_quintic_test.cc -o /tmp/playback_quintic_test
#include "../src/PlaybackQuintic.hh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>

static_assert(std::is_same_v<gp7::State, std::array<double, 8>>);
static_assert(std::tuple_size<gp7::State>::value == 8);

namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const std::string &label) {
  const double tolerance = 2e-10 * std::max(1.0, std::abs(expected));
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          label + ": actual=" + std::to_string(actual) +
          " expected=" + std::to_string(expected));
}

void compare(const gp7::PlaybackSample &actual,
             const gp7::PlaybackSample &expected) {
  for (std::size_t i = 0; i < 8; ++i) {
    near(actual.q[i], expected.q[i], "q[" + std::to_string(i) + "]");
    near(actual.v[i], expected.v[i], "v[" + std::to_string(i) + "]");
    near(actual.a[i], expected.a[i], "a[" + std::to_string(i) + "]");
  }
}

// Eight independent, nontrivial degree-five polynomials in absolute seconds.
// Compute derivatives by a separate power-series sum rather than the helper's
// normalized Hermite/Horner implementation.
gp7::PlaybackSample polynomial(double time) {
  gp7::PlaybackSample value;
  constexpr std::array<double, 6> base{0.31, -0.73, 0.19, -0.27, 0.13, -0.041};
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t k = 0; k < 6; ++k) {
      const double c = base[k] * (1.0 + 0.17 * static_cast<double>(i)) +
                       0.003 * static_cast<double>(i * (k + 1));
      value.q[i] += c * std::pow(time, static_cast<int>(k));
      if (k >= 1)
        value.v[i] += static_cast<double>(k) * c *
                      std::pow(time, static_cast<int>(k - 1));
      if (k >= 2)
        value.a[i] += static_cast<double>(k * (k - 1)) * c *
                      std::pow(time, static_cast<int>(k - 2));
    }
  }
  return value;
}

template <class F> void invalid(F operation) {
  bool threw = false;
  try { operation(); }
  catch (const std::invalid_argument &) { threw = true; }
  require(threw, "expected invalid_argument");
}

void endpoints() {
  gp7::PlaybackSample left, right;
  for (std::size_t i = 0; i < 8; ++i) {
    const double n = static_cast<double>(i);
    left.q[i] = n + 0.4; left.v[i] = -n - 0.3; left.a[i] = n + 0.2;
    right.q[i] = -n - 0.7; right.v[i] = n + 0.6; right.a[i] = -n - 0.5;
  }
  for (double h : {0.007, 0.31, 2.7}) {
    const auto start = gp7::interpolate(left, right, h, 0.0);
    const auto end = gp7::interpolate(left, right, h, h);
    require(start.q == left.q && start.v == left.v && start.a == left.a,
            "left endpoint must be exact");
    require(end.q == right.q && end.v == right.v && end.a == right.a,
            "right endpoint must be exact");
    // Also exercise the actual polynomial arbitrarily near each knot.
    // Compare to the same interval's exact endpoint Taylor limit. A tiny
    // interval has large jerk, so use adjacent representable times, not an
    // arbitrary relative time offset that changes acceleration appreciably.
    compare(gp7::interpolate(left, right, h, std::nextafter(0.0, h)), left);
    // At the right knot, finite time resolution times O(h^-3) jerk can
    // exceed the reconstruction tolerance for the shortest interval.
    const auto almost_end = gp7::interpolate(left, right, h, std::nextafter(h, 0.0));
    for (std::size_t i = 0; i < 8; ++i) {
      near(almost_end.q[i], right.q[i], "near-end q");
      near(almost_end.v[i], right.v[i], "near-end v");
      require(std::abs(almost_end.a[i] - right.a[i]) < 1e-7,
              "near-end a must approach endpoint acceleration");
    }
  }
}

void reconstruction() {
  constexpr std::array<double, 6> knots{-0.4, -0.33, 0.12, 1.42, 1.61, 2.4};
  for (std::size_t interval = 1; interval < knots.size(); ++interval) {
    const double origin = knots[interval - 1];
    const double h = knots[interval] - origin;
    const auto left = polynomial(origin), right = polynomial(knots[interval]);
    for (int step = 0; step <= 100; ++step) {
      const double t = step == 100 ? h : h * static_cast<double>(step) / 100.0;
      compare(gp7::interpolate(left, right, h, t), polynomial(origin + t));
    }
    compare(gp7::interpolate(left, right, h, h / 2.0),
            polynomial(origin + h / 2.0));
  }
}

void rest_to_rest_shape() {
  gp7::PlaybackSample left, right;
  right.q.fill(1.0);
  const auto mid = gp7::interpolate(left, right, 2.0, 1.0);
  for (std::size_t i = 0; i < 8; ++i) {
    near(mid.q[i], 0.5, "unit midpoint position");
    near(mid.v[i], 0.9375, "unit midpoint velocity in seconds");
    near(mid.a[i], 0.0, "unit midpoint acceleration");
  }
}

void invalid_inputs() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const gp7::PlaybackSample zero;
  for (double h : {0.0, -1.0, nan, inf, -inf})
    invalid([&] { gp7::interpolate(zero, zero, h, 0.0); });
  for (double t : {-0.1, 1.1, nan, inf, -inf})
    invalid([&] { gp7::interpolate(zero, zero, 1.0, t); });
  for (double bad : {nan, inf, -inf}) {
    for (std::size_t i = 0; i < 8; ++i) {
      for (auto member : {&gp7::PlaybackSample::q, &gp7::PlaybackSample::v,
                          &gp7::PlaybackSample::a}) {
        auto broken = zero;
        (broken.*member)[i] = bad;
        // Validate both endpoints even when evaluation returns the other knot.
        for (double t : {0.0, 0.5, 1.0}) {
          invalid([&] { gp7::interpolate(broken, zero, 1.0, t); });
          invalid([&] { gp7::interpolate(zero, broken, 1.0, t); });
        }
      }
    }
  }
  auto right = zero;
  right.q.fill(1.0);
  bool overflowed = false;
  try { gp7::interpolate(zero, right, 1e-200, 2.5e-201); }
  catch (const std::overflow_error &) { overflowed = true; }
  require(overflowed, "unrepresentable derivatives must throw");
}
}  // namespace

int main() {
  try {
    endpoints();
    std::cout << "PASS endpoint q/v/a (including near-knot evaluation)\n";
    reconstruction();
    std::cout << "PASS degree-5 reconstruction: 8 axes, 5 unequal intervals, dense samples and midpoints\n";
    rest_to_rest_shape();
    std::cout << "PASS scalar shape8 and seconds-scaled analytical derivatives\n";
    invalid_inputs();
    std::cout << "PASS invalid/nonfinite inputs and output overflow\n";
    std::cout << "All playback quintic tests passed.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
