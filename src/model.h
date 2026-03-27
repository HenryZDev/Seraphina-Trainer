#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <string>

#include "types.h"

struct SparseInput
{
    // Holds indices of active features for each sample in the batch
    torch::Tensor indices;  // [batch_size, max_active]
    torch::Tensor counts;   // [batch_size]
    int64_t feature_size;
    int64_t max_active;
};

struct FeatureTransformerImpl : torch::nn::Module
{
    torch::Tensor weight;  // [out_features, in_features]
    torch::Tensor bias;    // [out_features]

    FeatureTransformerImpl(int64_t in_features, int64_t out_features)
    {
        weight = register_parameter("weight", torch::randn({out_features, in_features}));
        bias = register_parameter("bias", torch::zeros(out_features));
    }

    torch::Tensor forward(const SparseInput& input)
    {
        auto device = weight.device();
        auto dtype = weight.dtype();

        int64_t batch_size = input.indices.size(0);
        int64_t max_active = input.indices.size(1);

        // 1. Create a mask to identify which indices are active [batch_size, max_active]
        torch::Tensor range = torch::arange(max_active, input.indices.options().device(device));
        torch::Tensor mask = range.unsqueeze(0) < input.counts.to(device).unsqueeze(1);

        // 2. Sanitize indices and perform lookup [batch_size, max_active, out_features]
        torch::Tensor safe_indices = input.indices.to(device).clamp(0, weight.size(1) - 1);
        torch::Tensor values = torch::embedding(weight.t(), safe_indices);

        // 3. Apply mask and sum over active features dimension
        values = values * mask.unsqueeze(-1).to(dtype);
        torch::Tensor sum = values.sum(1);

        return sum + bias;
    }
};
TORCH_MODULE(FeatureTransformer);

struct SeraphinaNNUE : torch::nn::Module
{
    FeatureTransformer ft{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    SeraphinaNNUE(int feature = (32 * 12 * 64), int l1 = 3072, int l2 = 16, int l3 = 32, int output = 1)
    {
        ft = register_module("ft", FeatureTransformer(feature, l1 / 2));
        fc1 = register_module("fc1", torch::nn::Linear(l1, l2));
        fc2 = register_module("fc2", torch::nn::Linear(l2, l3));
        fc3 = register_module("fc3", torch::nn::Linear(l3, output));
    }

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

    torch::Tensor forward(const SparseInput& input1, const SparseInput& input2)
    {
        auto x1 = ft->forward(input1);
        auto x2 = ft->forward(input2);
        auto x = torch::cat({x1, x2}, 1);
        x = fc1->forward(x);
        x = torch::clamp(x, 0, 1);
        x = fc2->forward(x);
        x = torch::clamp(x, 0, 1);
        x = fc3->forward(x);
        return x;
    }

    void save(std::string path, int q1 = 64, int q2 = 64);
};

void save_torch_weights(SeraphinaNNUE& nnue, std::string path);
void load_torch_weights(SeraphinaNNUE& nnue, std::string path);