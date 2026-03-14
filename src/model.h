#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <string>

#include "types.h"

struct SeraphinaNNUE : torch::nn::Module
{
    SeraphinaNNUE(int feature = (32 * 12 * 64), int l1 = 3072, int l2 = 16, int l3 = 32, int output = 1)
    {
        ft = register_module("ft", torch::nn::Linear(feature, l1 / 2));
        fc1 = register_module("fc1", torch::nn::Linear(l1, l2));
        fc2 = register_module("fc2", torch::nn::Linear(l2, l3));
        fc3 = register_module("fc3", torch::nn::Linear(l3, output));
    }

    torch::nn::Linear ft{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    static inline int king_square_index(int relative_king_square)
    {
        // clang-format off
        constexpr int indices[SQ_NUM]
        {
            -1, -1, -1, -1, 31, 30, 29, 28,
            -1, -1, -1, -1, 27, 26, 25, 24,
            -1, -1, -1, -1, 23, 22, 21, 20,
            -1, -1, -1, -1, 19, 18, 17, 16,
            -1, -1, -1, -1, 15, 14, 13, 12,
            -1, -1, -1, -1, 11, 10, 9, 8,
            -1, -1, -1, -1, 7, 6, 5, 4,
            -1, -1, -1, -1, 3, 2, 1, 0,
        };
        // clang-format on

        return indices[relative_king_square];
    }

    torch::Tensor forward(torch::Tensor& x1, torch::Tensor& x2);
    void save(std::string path, int q1 = 64, int q2 = 64);
};

void save_torch_weights(SeraphinaNNUE& nnue, std::string path);
void load_torch_weights(SeraphinaNNUE& nnue, std::string path);