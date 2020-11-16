#include "polynomials.h"
#include "buchberger.h"

#include <iostream>

int main() {

  Polynomial f1 = {{1, {1,0,0,0,0,0,0,0}},
		   {1, {0,1,0,0,0,0,0,0}},
		   {1, {0,0,1,0,0,0,0,0}},
		   {1, {0,0,0,1,0,0,0,0}},
		   {1, {0,0,0,0,1,0,0,0}},
		   {1, {0,0,0,0,0,1,0,0}}};
  Polynomial f2 = {{1, {1,1,0,0,0,0,0,0}},
		   {1, {0,1,1,0,0,0,0,0}},
		   {1, {0,0,1,1,0,0,0,0}},
		   {1, {0,0,0,1,1,0,0,0}},
		   {1, {0,0,0,0,1,1,0,0}},
		   {1, {1,0,0,0,0,1,0,0}}};
  Polynomial f3 = {{1, {1,1,1,0,0,0,0,0}},
		   {1, {0,1,1,1,0,0,0,0}},
		   {1, {0,0,1,1,1,0,0,0}},
		   {1, {0,0,0,1,1,1,0,0}},
		   {1, {1,0,0,0,1,1,0,0}},
		   {1, {1,1,0,0,0,1,0,0}}};
  Polynomial f4 = {{1, {1,1,1,1,0,0,0,0}},
		   {1, {0,1,1,1,1,0,0,0}},
		   {1, {0,0,1,1,1,1,0,0}},
		   {1, {1,0,0,1,1,1,0,0}},
		   {1, {1,1,0,0,1,1,0,0}},
		   {1, {1,1,1,0,0,1,0,0}}};
  Polynomial f5 = {{1, {1,1,1,1,1,0,0,0}},
		   {1, {0,1,1,1,1,1,0,0}},
		   {1, {1,0,1,1,1,1,0,0}},
		   {1, {1,1,0,1,1,1,0,0}},
		   {1, {1,1,1,0,1,1,0,0}},
		   {1, {1,1,1,1,0,1,0,0}}};
  Polynomial f6 = {{1, {1,1,1,1,1,1,0,0}},
		   {-1, {0,0,0,0,0,0,0,0}}};

  std::vector<Polynomial> F = {f1, f2, f3, f4, f5, f6};

  auto G = buchberger(F);
  G = minimalize(G);
  G = interreduce(G);

  std::cout << "\n" << G.size() << "\n\n";
  for (auto g : G)
    std::cout << g << "\n";
  std::cout << std::endl;
}