#pragma once

#include <string>
#include <thread>
#include <vector>

#include "argparse.hpp"
#include "model.h"
#include "terminal.h"

namespace binpack
{
namespace binpack
{
struct TrainingDataEntry;
}
}  // namespace binpack

namespace binpackloader
{
using DataEntry = binpack::binpack::TrainingDataEntry;
using DataSet = std::vector<DataEntry>;
}  // namespace binpackloader

class Trainer
{
    bool unix = get_unix();
    bool cuda = true;
    torch::Tensor x1_sparse, x2_sparse, target;

public:
    void setup_inputs_and_outputs(const binpackloader::DataSet& positions, float& sigmoid_scale, float& start_lambda,
                                  float& end_lambda, int& current_epoch, int& max_epochs, int& threads);
    /*
    void train(argparse::ArgumentParser& program, SeraphinaNNUE& nnue,
            std::vector<std::string>& train_files,
            std::vector<std::string>& val_files,
            const std::string& output_dir,
            float lr = 0.001, int lr_drop_ep = 1, float lr_drop_ratio = 0.992,
            float start_lambda = 1.0, float end_lambda = 0.7,
            int batch_size = 16384, int max_epochs = 800,
            int epoch_size = 1e8, int val_epoch_size = 1e7,
            int threads = std::thread::hardware_concurrency(),
            int random_fen_skipping = 0, int early_fen_skipping = 16,
            float sigmoid_scale = 1.0 / 160.0,
            int save_rate = 10,
            std::string optim = "muon");
    */
    void train(SeraphinaNNUE& nnue, std::vector<std::string>& train_files, std::vector<std::string>& val_files,
               const std::string& output_dir, float lr = 0.001, int lr_drop_ep = 1, float lr_drop_ratio = 0.992,
               float start_lambda = 1.0, float end_lambda = 0.7, int batch_size = 16384, int max_epochs = 800,
               int epoch_size = 1e8, int val_epoch_size = 1e7, int threads = std::thread::hardware_concurrency(),
               int random_fen_skipping = 0, int early_fen_skipping = 16, float sigmoid_scale = 1.0 / 160.0,
               int save_rate = 10, std::string optim = "muon", bool resume = false, std::string resume_network = "");
    Trainer()
    {
        if (!torch::cuda::is_available())
            cuda = false;
    };
};