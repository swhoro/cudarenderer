#pragma once

#include "Vec.hpp"
#include <vector>

template <int dim>
std::vector<Vec<dim>> bacthVecAdd(std::vector<Vec<dim>> &a, std::vector<Vec<dim>> &b);
