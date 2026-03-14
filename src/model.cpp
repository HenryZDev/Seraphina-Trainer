#include "model.h"

#include <fstream>
#include <sstream>

torch::Tensor SeraphinaNNUE::forward(torch::Tensor& x1, torch::Tensor& x2)
{
    auto device = ft->weight.device();
    x1 = ft->forward(x1);
    x2 = ft->forward(x2);

    torch::Tensor x = torch::cat({x1, x2}, 1).to(device);
    x = fc1->forward(x);
    x = torch::clamp(x, 0, 1);
    x = fc2->forward(x);
    x = torch::clamp(x, 0, 1);
    x = fc3->forward(x);

    return x;
}

void SeraphinaNNUE::save(std::string path, int q1, int q2)
{
    std::ofstream out(path, std::ios::binary);

    auto process_and_write_ft = [&](torch::nn::Linear layer) {
        at::Tensor w = layer->weight.detach().cpu() * q1;
        at::Tensor w_clamped =
            torch::clamp(w, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()).to(torch::kInt16);
        out.write(reinterpret_cast<const char*>(w_clamped.data_ptr<int16_t>()), w_clamped.numel() * sizeof(int16_t));

        at::Tensor b = layer->bias.detach().cpu() * q2;
        at::Tensor b_clamped =
            torch::clamp(b, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()).to(torch::kInt16);
        out.write(reinterpret_cast<const char*>(b_clamped.data_ptr<int16_t>()), b_clamped.numel() * sizeof(int16_t));
    };

    auto process_and_write_fc = [&](torch::nn::Linear layer) {
        at::Tensor w = layer->weight.detach().cpu() * q1;
        at::Tensor w_clamped =
            torch::clamp(w, std::numeric_limits<int8_t>::min(), std::numeric_limits<int32_t>::max()).to(torch::kInt8);
        out.write(reinterpret_cast<const char*>(w_clamped.data_ptr<int8_t>()), w_clamped.numel() * sizeof(int8_t));

        at::Tensor b = layer->bias.detach().cpu() * q2;
        at::Tensor b_clamped =
            torch::clamp(b, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()).to(torch::kInt32);
        out.write(reinterpret_cast<const char*>(b_clamped.data_ptr<int32_t>()), b_clamped.numel() * sizeof(int32_t));
    };

    process_and_write_ft(ft);
    process_and_write_fc(fc1);
    process_and_write_fc(fc2);
    process_and_write_fc(fc3);

    out.close();
}

void save_torch_weights(SeraphinaNNUE& nnue, std::string path)
{
    std::stringstream ss;
    torch::save(nnue.parameters(), ss);
    std::ofstream out(path, std::ios::binary);
    out << ss.rdbuf();
    out.close();
}

void load_torch_weights(SeraphinaNNUE& nnue, std::string path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    in.close();
    torch::load(nnue.parameters(), ss);
}