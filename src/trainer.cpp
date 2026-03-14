#include "trainer.h"

#include <omp.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "binpack/binpackloader.h"
#include "muon.h"
#include "progressbar.h"

using BatchLoader = binpackloader::BinpackLoader;

static inline int index(int piece_square, binpack::chess::Piece& piece, int king_square, Seraphina::Color view)
{
    const std::uint8_t piece_type = static_cast<uint8_t>(piece.type());
    const std::uint8_t piece_color = static_cast<uint8_t>(piece.color());

    piece_square ^= 56;
    king_square ^= 56;

    const int oP = piece_type + 6 * (piece_color != view);
    const int oK = (7 * !(king_square & 4)) ^ (56 * view) ^ king_square;
    const int oSq = (7 * !(king_square & 4)) ^ (56 * view) ^ piece_square;

    return SeraphinaNNUE::king_square_index(oK) * 12 * 64 + oP * 64 + oSq;
}

void Trainer::setup_inputs_and_outputs(const binpackloader::DataSet& positions, float& sigmoid_scale,
                                       float& start_lambda, float& end_lambda, int& current_epoch, int& max_epochs,
                                       int& threads)
{
    torch::set_num_threads(1);
    int batch_size = positions.size();
    target = torch::zeros({batch_size, 1});

    std::vector<std::vector<int>> x1_indices(batch_size);
    std::vector<std::vector<int>> x2_indices(batch_size);

    // Prepare per-thread storage
    std::vector<std::vector<int64_t>> x1_row_indices_threads(threads);
    std::vector<std::vector<int64_t>> x1_col_indices_threads(threads);
    std::vector<std::vector<float>> x1_values_threads(threads);
    std::vector<std::vector<int64_t>> x2_row_indices_threads(threads);
    std::vector<std::vector<int64_t>> x2_col_indices_threads(threads);
    std::vector<std::vector<float>> x2_values_threads(threads);

#pragma omp parallel for schedule(static, 64) num_threads(threads)
    for (int b = 0; b < batch_size; ++b)
    {
        int tid = omp_get_thread_num();
        const binpack::chess::Position& pos = positions[b].pos;
        binpack::chess::Square wKingSq = pos.kingSquare(binpack::chess::Color::White);
        binpack::chess::Square bKingSq = pos.kingSquare(binpack::chess::Color::Black);
        binpack::chess::Bitboard pieces = pos.piecesBB();

        for (binpack::chess::Square& sq : pieces)
        {
            binpack::chess::Piece piece = pos.pieceAt(sq);
            int piece_index_white_pov =
                index(static_cast<int>(sq), piece, static_cast<int>(wKingSq), Seraphina::Color::WHITE);
            int piece_index_black_pov =
                index(static_cast<int>(sq), piece, static_cast<int>(bKingSq), Seraphina::Color::BLACK);

            if (pos.sideToMove() == binpack::chess::Color::White)
            {
                x1_indices[b].emplace_back(piece_index_white_pov);
                x2_indices[b].emplace_back(piece_index_black_pov);
            } else
            {
                x2_indices[b].emplace_back(piece_index_white_pov);
                x1_indices[b].emplace_back(piece_index_black_pov);
            }
        }

        for (int& idx : x1_indices[b])
        {
            x1_row_indices_threads[tid].emplace_back(b);
            x1_col_indices_threads[tid].emplace_back(idx);
            x1_values_threads[tid].emplace_back(static_cast<float>(idx));
        }

        for (int& idx : x2_indices[b])
        {
            x2_row_indices_threads[tid].emplace_back(b);
            x2_col_indices_threads[tid].emplace_back(idx);
            x2_values_threads[tid].emplace_back(static_cast<float>(idx));
        }

        float p_value = positions[b].score;
        float w_value = positions[b].result;
        float p_target = 1 / (1 + std::exp(-p_value * sigmoid_scale));
        float w_target = (w_value + 1) / 2.0f;
        float actual_lambda = start_lambda + (end_lambda - start_lambda) * (current_epoch / float(max_epochs));
        target[b][0] = (actual_lambda * p_target + (1.0f - actual_lambda) * w_target);
    }

    // Merge per-thread vectors into the main vectors in order
    std::vector<int64_t> x1_row_indices, x1_col_indices, x2_row_indices, x2_col_indices;
    std::vector<float> x1_values, x2_values;

    for (int t = 0; t < threads; ++t)
    {
        x1_row_indices.insert(x1_row_indices.end(), x1_row_indices_threads[t].begin(), x1_row_indices_threads[t].end());
        x1_col_indices.insert(x1_col_indices.end(), x1_col_indices_threads[t].begin(), x1_col_indices_threads[t].end());
        x1_values.insert(x1_values.end(), x1_values_threads[t].begin(), x1_values_threads[t].end());
        x2_row_indices.insert(x2_row_indices.end(), x2_row_indices_threads[t].begin(), x2_row_indices_threads[t].end());
        x2_col_indices.insert(x2_col_indices.end(), x2_col_indices_threads[t].begin(), x2_col_indices_threads[t].end());
        x2_values.insert(x2_values.end(), x2_values_threads[t].begin(), x2_values_threads[t].end());
    }

    at::Tensor x1_row_tensor = torch::tensor(x1_row_indices, torch::kInt64);
    at::Tensor x1_col_tensor = torch::tensor(x1_col_indices, torch::kInt64);
    at::Tensor x1_indices_tensor = torch::stack({x1_row_tensor, x1_col_tensor}, 0);
    at::Tensor x1_values_tensor = torch::tensor(x1_values, torch::kFloat);
    x1_sparse = torch::sparse_coo_tensor(x1_indices_tensor, x1_values_tensor, {batch_size, 32 * 12 * 64});

    at::Tensor x2_row_tensor = torch::tensor(x2_row_indices, torch::kInt64);
    at::Tensor x2_col_tensor = torch::tensor(x2_col_indices, torch::kInt64);
    at::Tensor x2_indices_tensor = torch::stack({x2_row_tensor, x2_col_tensor}, 0);
    at::Tensor x2_values_tensor = torch::tensor(x2_values, torch::kFloat);
    x2_sparse = torch::sparse_coo_tensor(x2_indices_tensor, x2_values_tensor, {batch_size, 32 * 12 * 64});

    if (cuda)
    {
        x1_sparse = x1_sparse.to(torch::kCUDA);
        x2_sparse = x2_sparse.to(torch::kCUDA);
        target = target.to(torch::kCUDA);
    } else
    {
        x1_sparse = x1_sparse.to(torch::kCPU);
        x2_sparse = x2_sparse.to(torch::kCPU);
        target = target.to(torch::kCPU);
    }

    torch::set_num_threads(24);
}

void Trainer::train(SeraphinaNNUE& nnue, std::vector<std::string>& train_files, std::vector<std::string>& val_files,
                    const std::string& output_dir, float lr, int lr_drop_ep, float lr_drop_ratio, float start_lambda,
                    float end_lambda, int batch_size, int max_epochs, int epoch_size, int val_epoch_size, int threads,
                    int random_fen_skipping, int early_fen_skipping, float sigmoid_scale, int save_rate,
                    std::string optim, bool resume, std::string resume_network)
{
    torch::globalContext().setBenchmarkCuDNN(true);

    if (cuda)
    {
        nnue.to(torch::kCUDA);
        std::cout << Blue(unix) << "Model is allocated on " << Green(unix) << "CUDA\n" << Vanilla(unix);
    } else
    {
        nnue.to(torch::kCPU);
        std::cout << Blue(unix) << "Model is allocated on " << Yellow(unix) << "CPU\n" << Vanilla(unix);
    }

    std::transform(optim.begin(), optim.end(), optim.begin(), tolower);
    Muon muon(nnue.parameters(), MuonOptions{.lr = lr});
    torch::optim::Adam adam(nnue.parameters(), torch::optim::AdamOptions(lr).betas({0.95, 0.999}).eps(1e-8));
    torch::optim::AdamW adamw(nnue.parameters(),
                              torch::optim::AdamWOptions(lr).betas({0.95, 0.999}).eps(1e-8).weight_decay(1e-4));
    torch::nn::MSELoss loss_fn;
    Progress progress;
    std::stringstream path;

    std::cout << Blue(unix) << "Using " << Vanilla(unix) << Bold(unix) << optim << Blue(unix) << " optimizer\n"
              << Vanilla(unix);

    binpackloader::BinpackLoader train_loader{train_files, batch_size, threads, early_fen_skipping,
                                              random_fen_skipping};
    train_loader.start();

    std::optional<binpackloader::BinpackLoader> val_loader;
    if (val_files.size() > 0)
    {
        val_loader.emplace(val_files, batch_size, threads, early_fen_skipping, random_fen_skipping);
        val_loader->start();
    }

    /*
    if (std::optional<std::string> previous = program.present("--resume"))
    {
        load_torch_weights(nnue, previous.value());
        std::cout << Blue(unix) << "Resume from previous " << previous.value() << "\n";
    }
    */

    if (resume)
    {
        if (resume_network.empty())
        {
            std::cerr << Red(unix) << "Error: resume_network is empty\n" << Vanilla(unix);
        }

        load_torch_weights(nnue, resume_network);
    }

    if (optim == "muon")
    {
        for (int epoch = 1; epoch <= max_epochs; ++epoch)
        {
            torch::Tensor batch_loss, val_batch_loss;
            torch::Tensor total_epoch_loss = torch::zeros({}, nnue.parameters().front().options());
            float total_val_loss = 0.0;
            float itps = 0.0;

            if (epoch % lr_drop_ep == 0 && epoch > 0)
            {
                lr *= lr_drop_ratio;
                muon.set_lr(lr);
            }

            int num_batches = epoch_size / train_loader.batch_size;
            for (int b = 1; b <= num_batches; ++b)
            {
                std::chrono::steady_clock::time_point batch_start = std::chrono::high_resolution_clock::now();

                binpackloader::DataSet ds = train_loader.next();
                if (ds.size() == 0)
                {
                    std::cerr << "Warning: train_loader.next() returned an empty dataset at batch " << b << std::endl;
                    continue;  // Skip this batch
                }
                setup_inputs_and_outputs(ds, sigmoid_scale, start_lambda, end_lambda, epoch, max_epochs, threads);
                batch_loss = loss_fn(nnue.forward(x1_sparse, x2_sparse), target);
                total_epoch_loss += batch_loss;
                batch_loss.backward();

                std::chrono::steady_clock::time_point batch_end = std::chrono::high_resolution_clock::now();
                float batch_time = std::chrono::duration<float>(batch_end - batch_start).count();
                itps = 1.0 / batch_time;

                progress.update(epoch, max_epochs, b, epoch_size / train_loader.batch_size, batch_loss.item<float>(),
                                (total_epoch_loss / b).item<float>(), itps);

                muon.step();
                nnue.zero_grad();
            }

            if (val_loader.has_value())
            {
                torch::NoGradGuard no_grad;

                for (int b = 1; b <= val_epoch_size / val_loader->batch_size; ++b)
                {
                    auto ds = val_loader->next();
                    setup_inputs_and_outputs(ds, sigmoid_scale, start_lambda, end_lambda, epoch, max_epochs, threads);

                    val_batch_loss = loss_fn(nnue.forward(x1_sparse, x2_sparse), target);
                    total_val_loss += val_batch_loss.item<float>();
                }
            }

            total_val_loss /= (val_epoch_size / val_loader->batch_size);
            float epoch_loss = (total_epoch_loss / num_batches).item<float>();
            progress.end_epoch(epoch, max_epochs, batch_loss.item<float>(), epoch_loss, itps, total_val_loss);

            if (epoch % save_rate == 0)
            {
                path << output_dir << "/epoch-" << epoch << ".nnue";
                nnue.save(path.str());
                save_torch_weights(nnue, path.str());
                path.clear();
            }
        }
    } else if (optim == "adam")
    {
        torch::optim::StepLR scheduler(adam, lr_drop_ep, lr_drop_ratio);

        for (int epoch = 1; epoch <= max_epochs; ++epoch)
        {
            torch::Tensor batch_loss, val_batch_loss;
            torch::Tensor total_epoch_loss = torch::zeros({}, nnue.parameters().front().options());
            float total_val_loss = 0.0;
            float itps = 0.0;

            int num_batches = epoch_size / train_loader.batch_size;
            for (int b = 1; b <= num_batches; ++b)
            {
                std::chrono::steady_clock::time_point batch_start = std::chrono::high_resolution_clock::now();

                binpackloader::DataSet ds = train_loader.next();
                setup_inputs_and_outputs(ds, sigmoid_scale, start_lambda, end_lambda, epoch, max_epochs, threads);
                batch_loss = loss_fn(nnue.forward(x1_sparse, x2_sparse), target);
                total_epoch_loss += batch_loss;
                batch_loss.backward();

                std::chrono::steady_clock::time_point batch_end = std::chrono::high_resolution_clock::now();
                float batch_time = std::chrono::duration<float>(batch_end - batch_start).count();
                itps = 1.0 / batch_time;

                progress.update(epoch, max_epochs, b, epoch_size / train_loader.batch_size, batch_loss.item<float>(),
                                (total_epoch_loss / b).item<float>(), itps);

                adam.step();
                nnue.zero_grad();
            }

            scheduler.step();

            if (val_loader.has_value())
            {
                torch::NoGradGuard no_grad;

                for (int b = 1; b <= val_epoch_size / val_loader->batch_size; ++b)
                {
                    auto ds = val_loader->next();
                    setup_inputs_and_outputs(ds, sigmoid_scale, start_lambda, end_lambda, epoch, max_epochs, threads);

                    val_batch_loss = loss_fn(nnue.forward(x1_sparse, x2_sparse), target);
                    total_val_loss += val_batch_loss.item<float>();
                }
            }

            total_val_loss /= (val_epoch_size / val_loader->batch_size);
            float epoch_loss = (total_epoch_loss / num_batches).item<float>();
            progress.end_epoch(epoch, max_epochs, batch_loss.item<float>(), epoch_loss, itps, total_val_loss);

            if (epoch % save_rate == 0)
            {
                path << output_dir << "/epoch-" << epoch << ".nnue";
                nnue.save(path.str());
                save_torch_weights(nnue, path.str());
                path.clear();
            }
        }
    } else
    {
        torch::optim::StepLR scheduler(adamw, lr_drop_ep, lr_drop_ratio);

        for (int epoch = 1; epoch <= max_epochs; ++epoch)
        {
            torch::Tensor batch_loss, val_batch_loss;
            torch::Tensor total_epoch_loss = torch::zeros({}, nnue.parameters().front().options());
            float total_val_loss = 0.0;
            float itps = 0.0;

            int num_batches = epoch_size / train_loader.batch_size;
            for (int b = 1; b <= num_batches; ++b)
            {
                std::chrono::steady_clock::time_point batch_start = std::chrono::high_resolution_clock::now();

                binpackloader::DataSet ds = train_loader.next();
                setup_inputs_and_outputs(ds, sigmoid_scale, start_lambda, end_lambda, epoch, max_epochs, threads);
                batch_loss = loss_fn(nnue.forward(x1_sparse, x2_sparse), target);
                total_epoch_loss += batch_loss;
                batch_loss.backward();

                std::chrono::steady_clock::time_point batch_end = std::chrono::high_resolution_clock::now();
                float batch_time = std::chrono::duration<float>(batch_end - batch_start).count();
                itps = 1.0 / batch_time;

                progress.update(epoch, max_epochs, b, epoch_size / train_loader.batch_size, batch_loss.item<float>(),
                                (total_epoch_loss / b).item<float>(), itps);

                adamw.step();
                nnue.zero_grad();
            }

            scheduler.step();

            if (val_loader.has_value())
            {
                torch::NoGradGuard no_grad;

                for (int b = 1; b <= val_epoch_size / val_loader->batch_size; ++b)
                {
                    auto ds = val_loader->next();
                    setup_inputs_and_outputs(ds, sigmoid_scale, start_lambda, end_lambda, epoch, max_epochs, threads);

                    val_batch_loss = loss_fn(nnue.forward(x1_sparse, x2_sparse), target);
                    total_val_loss += val_batch_loss.item<float>();
                }
            }

            total_val_loss /= (val_epoch_size / val_loader->batch_size);
            float epoch_loss = (total_epoch_loss / num_batches).item<float>();
            progress.end_epoch(epoch, max_epochs, batch_loss.item<float>(), epoch_loss, itps, total_val_loss);

            if (epoch % save_rate == 0)
            {
                path << output_dir << "/epoch-" << epoch << ".nnue";
                nnue.save(path.str());
                save_torch_weights(nnue, path.str());
                path.clear();
            }
        }
    }
}