#pragma once
#include <torch/torch.h>

#include <cmath>
#include <tuple>
#include <unordered_map>
#include <vector>

struct MuonOptions
{
    double lr = 1e-3;
    double weight_decay = 1e-4;
    double momentum = 0.95;
    bool nesterov = true;
    std::tuple<double, double, double> ns_coefficients = {3.4445, -4.7750, 2.0315};
    double eps = 1e-7;
    int ns_steps = 5;
};

struct MuonParamState
{
    int64_t step = 0;
    torch::Tensor momentum_buffer;
};

class Muon
{
public:
    Muon(std::vector<torch::Tensor> params, MuonOptions opts = {}) : params_(std::move(params)), options_(opts)
    {
        for (auto& p : params_)
        {
            MuonParamState st;
            st.momentum_buffer = torch::zeros_like(p);
            state_[p.unsafeGetTensorImpl()] = st;
        }
    }

    void set_lr(double lr) { options_.lr = lr; }

    void step()
    {
        torch::NoGradGuard no_grad;
        for (auto& p : params_)
        {
            if (!p.grad().defined())
                continue;
            auto grad = p.grad();
            if (grad.dim() != 2)
                continue;

            auto& st = state_[p.unsafeGetTensorImpl()];
            st.step++;

            // Momentum update
            st.momentum_buffer = options_.momentum * st.momentum_buffer + (1.0 - options_.momentum) * grad;
            auto update = options_.nesterov ? grad + options_.momentum * st.momentum_buffer : st.momentum_buffer;

            // Normalize before orthogonalization
            update = update / (update.norm().clamp(options_.eps));

            // Newton–Schulz orthogonalization
            update = zeropower_via_newtonschulz(update);

            // Effective learning rate (no aggressive scaling)
            double adjusted_lr = options_.lr;

            // Weight decay
            if (options_.weight_decay != 0)
                p.mul_(1.0 - adjusted_lr * options_.weight_decay);

            // Parameter update
            p.add_(update, -adjusted_lr);
        }
    }

private:
    torch::Tensor zeropower_via_newtonschulz(torch::Tensor grad)
    {
        auto [a, b, c] = options_.ns_coefficients;
        auto ortho_grad = grad.to(torch::kFloat32);
        if (grad.size(0) > grad.size(1))
            ortho_grad = ortho_grad.t();

        for (int i = 0; i < options_.ns_steps; ++i)
        {
            auto gram = ortho_grad.mm(ortho_grad.t());
            auto gram_update = torch::addmm(gram, gram, gram, b, c);
            ortho_grad = torch::addmm(ortho_grad, gram_update, ortho_grad, a);
        }

        if (grad.size(0) > grad.size(1))
            ortho_grad = ortho_grad.t();
        return ortho_grad;
    }

    std::vector<torch::Tensor> params_;
    MuonOptions options_;
    std::unordered_map<const c10::TensorImpl*, MuonParamState> state_;
};
