#pragma on

// turn off warnings for this
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "nnue_data_binpack_format.h"
#pragma GCC diagnostic pop

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

namespace binpackloader {

using DataEntry = binpack::binpack::TrainingDataEntry;
using DataSet   = std::vector<DataEntry>;
using binpack::binpack::CompressedTrainingDataEntryParallelReader;

// Data Filtering Strategy taken from Stockfish nnue pytorch
// https://github.com/official-stockfish/nnue-pytorch/blob/master/training_data_loader.cpp
std::function<bool(const DataEntry&)> make_skip_predicate(bool filtered,
                                                          int  random_fen_skipping,
                                                          bool wld_filtered,
                                                          int  early_fen_skipping) {
    if (filtered || random_fen_skipping || wld_filtered || early_fen_skipping) {
        return [random_fen_skipping,
                prob = static_cast<double>(random_fen_skipping) / (random_fen_skipping + 1),
                filtered,
                wld_filtered,
                early_fen_skipping](const DataEntry& e) {
            static constexpr int VALUE_NONE  = 32002;

            auto                 do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0
                                                    - e.score_result_prob() * e.score_result_prob());
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return e.isInCheck() || (e.isCapturingMove() && (e.score == 0 || e.seeGE(0)));
            };

            // Allow for predermined filtering without the need to remove positions from the dataset.
            if (e.score == VALUE_NONE)
                return true;

            if (e.ply <= early_fen_skipping) {
                return true;
            }

            if (random_fen_skipping && do_skip()) {
                return true;
            }

            if (filtered && do_filter())
                return true;

            if (wld_filtered && do_wld_skip())
                return true;

            return false;
        };
    }

    return nullptr;
}

/// @brief Multithreaded dataloader to load data in Stockfish's binpack format
struct BinpackLoader {

    static constexpr std::size_t                               ChunkSize = (1 << 22);

    std::vector<std::string>                                   paths;
    std::unique_ptr<CompressedTrainingDataEntryParallelReader> reader;

    std::vector<std::size_t>                                   permute_shuffle;
    DataSet                                                    buffer;
    DataSet                                                    active_buffer;
    DataSet                                                    active_batch;

    std::thread                                                readingThread;
    int                                                        batch_size;
    int                                                        current_batch_index  = 0;
    size_t                                                     total_positions_read = 0;
    int                                                        concurrency          = 8;

    std::function<bool(const DataEntry&)> skipPredicate;
    static constexpr auto openmode = std::ios::in | std::ios::binary;

    BinpackLoader(const std::vector<std::string>& filename,
                  int                             batch_size,
                  int                             concurrency,
                  int                             early_fen_skipping,
                  int                             random_fen_skipping)
        : batch_size(batch_size)
        , paths(filename)
        , concurrency(concurrency) {
        buffer.reserve(ChunkSize);
        active_buffer.reserve(ChunkSize);
        permute_shuffle.resize(ChunkSize);
        active_batch.reserve(batch_size);

        skipPredicate = make_skip_predicate(true, random_fen_skipping, true, early_fen_skipping);
        reader        = std::make_unique<binpack::binpack::CompressedTrainingDataEntryParallelReader>(
            concurrency,
            paths,
            openmode,
            false,
            skipPredicate);
    }

    void start() {

        current_batch_index = 0;

        shuffle();
        loadNext();
        loadToActiveBuffer();
        readingThread = std::thread(&BinpackLoader::loadNext, this);
    }

    void loadToActiveBuffer() {
        active_buffer.clear();
        for (int i = 0; i < buffer.size(); i++) {
            active_buffer.push_back(buffer[i]);
        }
    }

    void loadNext() {
        buffer.clear();

        auto k = reader->fill(buffer, ChunkSize);

        if (ChunkSize != k) {
            reader = std::make_unique<binpack::binpack::CompressedTrainingDataEntryParallelReader>(
                concurrency,
                paths,
                openmode,
                false,
                skipPredicate);
        }
    }

    DataSet& next() {
        active_batch.clear();

        for (int i = 0; i < batch_size; i++) {
            if (current_batch_index >= active_buffer.size()) {

                current_batch_index = 0;

                if (readingThread.joinable()) {
                    readingThread.join();
                }

                loadToActiveBuffer();
                shuffle();

                readingThread = std::thread(&BinpackLoader::loadNext, this);
            }

            active_batch.push_back(active_buffer[permute_shuffle[current_batch_index++]]);
        }

        return active_batch;
    }

    void shuffle() {
        std::iota(permute_shuffle.begin(), permute_shuffle.end(), 0);
        std::shuffle(permute_shuffle.begin(),
                     permute_shuffle.end(),
                     std::mt19937(std::random_device()()));
    }
};

}    // namespace binpackloader