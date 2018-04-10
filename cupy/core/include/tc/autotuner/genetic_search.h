/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <random>

#include "tc/autotuner/parameters.h"

namespace tc {
namespace autotune {

/**
 * GeneticSearch operates on generations of candidate solutions. Each
 * generation consists of multiple candidate solutions. Before a new
 * generation can be generated all candidates of the current one are assigned a
 * fitness value (1 / runtime (in us) in our case).
 *
 * The top candidates are preserved intact across generation (elitism).
 *
 * Each new candidate is generated with the following two mechanisms:
 *
 * crossover: 3 parent candidates are selected probabilistically (influenced by
 * their fitness, the higher it is the higher the chance of selection) and
 * merged into a new candidate
 *
 * mutation: parts of a candidate are randomly changed (mutated)
 *
 * The crossover rate controls how many candidates survive across generations
 * and how many new candidates are produced through crossover, e.g, 80%
 * crossover  rate would result in roughly 20% of the new generation to consist
 * of candidates that survived from the previous one. Note, that this is
 * different from elitism: elite candidates are not mutated but "survivors"
 * are.
 *
 * The mutation rate controls the probability with which mutation occurs.
 */

class GeneticSearch {
 public:
  /**
   * conf is used to determine which are the tunable parameters, the selected
   * values in conf are ignored and the population is randomized
   *
   * n is the population size
   *
   * crossoverRate is the probability ([0,100]) that a new candidate is produced
   * through reproduction
   *
   * mutationRate is the probability ([0,100]) that parameters are mutated
   * (randomly changed) whenever a new generation is created
   *
   * numberElites best candidates are preserved
   * across generations (elitism), number Elites must be less than n
   */
  GeneticSearch(
      const TuningConfiguration& conf,
      size_t n,
      uint8_t crossOverRate,
      uint8_t mutationRate,
      size_t numberElites);

  /**
   * confs are used to seed the first generation, the rest of the population is
   * randomly initialized
   *
   * n is the population size
   *
   * crossoverRate is the probability ([0,100]) that a new candidate is produced
   * through reproduction
   *
   * mutationRate is the probability ([0,100]) that parameters are mutated
   * (randomly changed) whenever a new generation is created
   *
   * numberElites best candidates are preserved
   * across generations (elitism), number Elites must be less than n
   */
  GeneticSearch(
      const std::vector<TuningConfiguration>& confs,
      size_t n,
      uint8_t crossOverRate,
      uint8_t mutationRate,
      size_t numberElites);

  void updateParameters();

 private:
  void breed();

  TuningConfiguration crossover(
      TuningConfiguration&,
      TuningConfiguration&,
      TuningConfiguration&) const;

 public:
  static constexpr int kMutateIterations = 1000;
  static constexpr int kMinCandidatesForBreeding = 3;

  using Population = std::vector<std::unique_ptr<CandidateConfiguration>>;

  Population population;
  TuningConfiguration lastBestConf;
  const size_t kMaxPopulationSize;
  const uint8_t kCrossOverRate;
  const uint8_t kMutationRate;
  const size_t kNumberElites;

  /*
   * c++11 seeding is (apparently) not of the highest quality:
   * http://www.pcg-random.org/posts/cpp-seeding-surprises.html
   * pcg-cpp seems like a good alternative:
   * https://www.johndcook.com/blog/2017/08/14/testing-rngs-with-practrand/
   * http://www.pcg-random.org/posts/pcg-passes-practrand.html
   */
  mutable std::mt19937_64 rng;
};

} // namespace autotune
} // namespace tc
