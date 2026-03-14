// #include "argparse.hpp"
#include "dataset.h"
#include "trainer.h"

static void env_check()
{
    bool unix = get_unix();

#if defined(_WIN64)
    std::cout << Blue(unix) << "Windows 64-bit | ";
#elif defined(_WIN32)
    std::cout << Blue(unix) << "Windows 32-bit | ";
#elif defined(__APPLE__)
    std::cout << Blue(unix) << "Apple OS | ";
#elif defined(__linux__)
    std::cout << Blue(unix) << "Linux | ";
#else
    std::cout << Red(unix) << "Unsupported OS" << Blue(unix) << " | ";
#endif

#if defined(_MSC_VER)
    std::cout << "Microsoft C++ (MSVC) "
              << (_MSC_VER >= 1900 ? ((float(_MSC_VER) - 500) / 100) : ((float(_MSC_VER) - 600) / 100)) << " | ";
#elif defined(__clang__)
    std::cout << "Clang " << __clang_version__ << " | ";
#elif defined(__GNUC__) && !defined(__GNUG__)
    std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << " | ";
#elif defined(__GNUC__) && defined(__GNUG__)
    std::cout << "G++ " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << " | ";
#endif

    std::cout << "CUDA ";

    if (torch::cuda::is_available())
    {
        std::cout << Green(unix) << "Available\n" << Vanilla(unix);
    } else
    {
        std::cout << Red(unix) << "Not Available\n" << Vanilla(unix);
    }
}

/**
 * @brief Counts total positions across all datasets from provided file paths.
 *
 * Iterates through file paths, reads headers, and sums positions for total count.
 *
 * @param files Vector of dataset file paths.
 * @return Total positions across all datasets.
 */
uint64_t count_total_positions(const std::vector<std::string>& files)
{
    uint64_t total_positions = 0;

    // Iterate through each file path and read dataset headers to count positions
    for (const std::string& path : files)
    {
        std::ifstream fin(path, std::ios::binary);
        dataset::DataSetHeader h{};
        fin.read(reinterpret_cast<char*>(&h), sizeof(dataset::DataSetHeader));
        total_positions += h.entry_count;
    }

    return total_positions;
}

/**
 * @brief Retrieves dataset file paths from the specified directory.
 *
 * Iterates through the directory and collects paths of dataset files.
 *
 * @param directory The directory containing dataset files.
 * @return Vector of dataset file paths.
 */
std::vector<std::string> fetch_dataset_paths(const std::string& directory)
{
    std::vector<std::string> files;

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory))
    {
        const std::string path = entry.path().string();
        files.emplace_back(path);
    }

    return files;
}

int main(int argc, char* argv[])
{
    env_check();

    std::vector<std::string> train_files = fetch_dataset_paths("/trainingdata");
    bool is_binpack = false;

    // Print training dataset file list if files are found
    if (!train_files.empty())
    {
        std::cout << "Training Dataset Files:" << std::endl;

        for (const std::string& file : train_files)
        {
            std::cout << file << std::endl;

            if (file.find(".binpack") != std::string::npos)
            {
                is_binpack = true;
            }
        }

        std::cout << "Total training files: " << train_files.size() << std::endl;

        // can't count total positions in binpack files
        if (!is_binpack)
        {
            std::cout << "Total training positions: " << count_total_positions(train_files) << std::endl << std::endl;
        }
    } else
    {
        std::cout << "No training files found in " << "/trainingdata" << std::endl << std::endl;
        exit(0);
    }

    // Fetch validation dataset paths
    std::vector<std::string> val_files;
    val_files = fetch_dataset_paths("/trainingdata");

    // Print validation dataset file list if files are found
    if (!val_files.empty())
    {
        std::cout << "Validation Dataset Files:" << std::endl;

        for (const std::string& file : val_files)
        {
            std::cout << file << std::endl;

            if (file.find(".binpack") != std::string::npos && !is_binpack)
            {
                std::cerr << "Validation dataset is binpack but training dataset is not. Exiting." << std::endl;
                exit(1);
            }
        }
        std::cout << "Total validation files: " << val_files.size() << std::endl;

        // can't count total positions in binpack files
        if (!is_binpack)
        {
            std::cout << "Total validation positions: " << count_total_positions(val_files) << std::endl << std::endl;
        }
    }

    const std::string output = "/nnue";
    constexpr float lr = 0.001;
    constexpr int lr_drop_epoch = 1;
    constexpr float lr_drop_ratio = 0.992;
    constexpr int batch_size = 16384;
    constexpr float startlambda = 1.0;
    constexpr float endlambda = 1.0;
    constexpr float total_epochs = 400;
    constexpr int epoch_size = 100000000;
    constexpr int val_epoch_size = 10000000;
    constexpr int binpackloader_concurrency = 24;
    constexpr float random_fen_skipping = 0;
    constexpr float early_fen_skipping = -1;
    constexpr float sigmoid_scale = 1.0 / 160.0;
    constexpr int save_rate = 10;
    constexpr std::string optimizer = "muon";
    constexpr bool resume = false;
    constexpr std::string resume_network = "";

    SeraphinaNNUE nnue(32 * 12 * 64, 1536 * 2);
    Trainer trainer;
    trainer.train(nnue, train_files, val_files, output, lr, lr_drop_epoch, lr_drop_ratio, startlambda, endlambda,
                  batch_size, total_epochs, epoch_size, val_epoch_size, binpackloader_concurrency, random_fen_skipping,
                  early_fen_skipping, sigmoid_scale, save_rate, optimizer, resume, resume_network);

    /*
        argparse::ArgumentParser program("SeraphinaTrainer");
    program.add_argument("--device")
        .default_value("cuda")
        .help("Device allocating nnue CUDA/CPU");
        program.add_argument("--data").required().help("Directory containing training files");
        program.add_argument("--val-data").help("Directory containing validation files");
    program.add_argument("--output").required().help("Output directory for network files");
    program.add_argument("--resume").help("Weights file to resume from");
    program.add_argument("--epochs")
        .default_value(800)
        .help("Total number of epochs to train for")
        .scan<'i', int>();
    program.add_argument("--concurrency")
        .default_value(24)
        .help("Sets the number of threads the sf binpack dataloader will use (if using the sf "
            "binpack dataloader.)")
        .scan<'i', int>();
    program.add_argument("--epoch-size")
        .default_value(100000000)
        .help("Total positions in each epoch")
        .scan<'i', int>();
    program.add_argument("--val-size")
        .default_value(10000000)
        .help("Total positions for each validation epoch")
        .scan<'i', int>();
    program.add_argument("--save-rate")
        .default_value(10)
        .help("How frequently to save quantized networks + weights")
        .scan<'i', int>();
    program.add_argument("--ft-size")
        .default_value(1536)
        .help("Number of neurons in the Feature Transformer")
        .scan<'i', int>();
    program.add_argument("--startlambda")
        .default_value(1.0f)
        .help("Ratio of evaluation at the start of the training (if applicable to the model being "
            "used)")
        .scan<'f', float>();
    program.add_argument("--endlambda")
        .default_value(0.7f)
        .help("Ratio of evaluation interpolated by the end of the training (if applicable to the "
            "model being used)")
        .scan<'f', float>();
    program.add_argument("--lr")
        .default_value(0.001f)
        .help("The starting learning rate for the optimizer")
        .scan<'f', float>();
    program.add_argument("--batch-size")
        .default_value(16384)
        .help("Number of positions in a mini-batch during training")
        .scan<'i', int>();
    program.add_argument("--lr-drop-epoch")
        .default_value(1)
        .help("Epoch to execute an LR drop at")
        .scan<'i', int>();
    program.add_argument("--lr-drop-ratio")
        .default_value(0.992f)
        .help("How much to scale down LR when dropping")
        .scan<'f', float>();
    program.add_argument("--sigmoid-scale")
        .default_value(1.0 / 160.0)
        .help("Sigmoid Scale for p_target scaling")
        .scan<'f', float>();
    program.add_argument("--optimizer")
        .default_value("muon")
        .help("Optimizer for training");
    program.add_argument("--skip").default_value(0).help("Skip fens randomly").scan<'i', int>();
    program.add_argument("--early-skip")
        .default_value(16)
        .help("Skip fens at the start of the training")
        .scan<'i', int>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    // Code below is from Grapheus
    // Fetch training dataset paths
    std::vector<std::string> train_files = fetch_dataset_paths(program.get("--data"));
    bool                     is_binpack = false;

    // Print training dataset file list if files are found
    if (!train_files.empty())
    {
        std::cout << "Training Dataset Files:" << std::endl;

        for (const std::string& file : train_files)
        {
            std::cout << file << std::endl;

            if (file.find(".binpack") != std::string::npos)
            {
                is_binpack = true;
            }
        }

        std::cout << "Total training files: " << train_files.size() << std::endl;

        // can't count total positions in binpack files
        if (!is_binpack)
        {
            std::cout << "Total training positions: " << count_total_positions(train_files)
                << std::endl
                << std::endl;
        }
    }
    else
    {
        std::cout << "No training files found in " << program.get("--data") << std::endl << std::endl;
        exit(0);
    }

    // Fetch validation dataset paths
    std::vector<std::string> val_files;

    if (program.present("--val-data"))
    {
        val_files = fetch_dataset_paths(program.get("--val-data"));
    }

    // Print validation dataset file list if files are found
    if (!val_files.empty())
    {
        std::cout << "Validation Dataset Files:" << std::endl;

        for (const std::string& file : val_files)
        {
            std::cout << file << std::endl;

            if (file.find(".binpack") != std::string::npos && !is_binpack)
            {
                std::cerr << "Validation dataset is binpack but training dataset is not. Exiting."
                    << std::endl;
                exit(1);
            }
        }
        std::cout << "Total validation files: " << val_files.size() << std::endl;

        // can't count total positions in binpack files
        if (!is_binpack)
        {
            std::cout << "Total validation positions: " << count_total_positions(val_files)
                << std::endl
                << std::endl;
        }
    }

    const std::string device = program.get("--device");
    const std::string output = program.get("--output");
    const int   total_epochs = program.get<int>("--epochs");
    const int   epoch_size = program.get<int>("--epoch-size");
    const int   val_epoch_size = program.get<int>("--val-size");
    const int   save_rate = program.get<int>("--save-rate");
    const int   ft_size = program.get<int>("--ft-size");
    const float startlambda = program.get<float>("--startlambda");
    const float endlambda = program.get<float>("--endlambda");
    const float lr = program.get<float>("--lr");
    const int   batch_size = program.get<int>("--batch-size");
    const int   lr_drop_epoch = program.get<int>("--lr-drop-epoch");
    const float lr_drop_ratio = program.get<float>("--lr-drop-ratio");
    const int   binpackloader_concurrency = program.get<int>("--concurrency");
    const float sigmoid_scale = program.get<float>("--sigmoid-scale");
    std::string optimizer = program.get("--optimizer");
    const int   random_fen_skipping = program.get<int>("--skip");
    const int   early_fen_skipping = program.get<int>("--early-skip");

    SeraphinaNNUE nnue(32 * 12 * 64, ft_size * 2);
        Trainer trainer(nnue, device);
    trainer.train(program, nnue,
        train_files, val_files, output,
        lr, lr_drop_epoch, lr_drop_ratio,
        startlambda, endlambda,
        batch_size, total_epochs,
        epoch_size, val_epoch_size,
        binpackloader_concurrency,
        random_fen_skipping, early_fen_skipping,
        sigmoid_scale, save_rate, optimizer);
     */

    return 0;
}