#pragma once

#include <chrono>
#include <iostream>

#include "terminal.h"

struct Progress
{
    int bar_width = 30;  // narrower bar to avoid wrapping
    bool unix = get_unix();

    void update(int epoch, int total_epochs, int batch_idx, int batches_in_epoch, float batch_loss, float epoch_loss,
                float itps)
    {
        float progress = static_cast<float>(batch_idx) / batches_in_epoch;
        if (progress > 1.0f)
            progress = 1.0f;
        int pos = static_cast<int>(progress * bar_width);

        std::ostringstream line;
        line << Blue(unix) << "Epoch " << epoch << "/" << total_epochs << " | "
             << "Batch " << batch_idx << "/" << batches_in_epoch << Vanilla(unix) << " [";

        for (int i = 0; i < bar_width; ++i)
        {
            if (i <= pos)
                line << WhiteBG(unix) << " " << Vanilla(unix);
            else
                line << Vanilla(unix) << " ";
        }

        line << "] " << Blue(unix) << static_cast<int>(progress * 100.0f) << "% | Batch Loss: " << std::fixed
             << std::setprecision(4) << batch_loss << " | Epoch Loss: " << std::fixed << std::setprecision(4)
             << epoch_loss << " | Speed: " << std::fixed << std::setprecision(2) << itps << " it/s" << Vanilla(unix);

        // Clear the line first, then overwrite
        std::cout << "\r" << std::string(120, ' ') << "\r" << line.str() << std::flush;
    }

    void end_epoch(int epoch, int total_epochs, float batch_loss, float epoch_loss, float itps, float val_loss)
    {
        std::ostringstream line;
        line << Blue(unix) << "Epoch " << epoch << "/" << total_epochs << Vanilla(unix) << " [" << WhiteBG(unix)
             << std::string(bar_width, ' ') << Vanilla(unix) << "]" << Blue(unix) << " 100%"
             << " | Batch Loss: " << std::fixed << std::setprecision(4) << batch_loss << " | Epoch Loss: " << std::fixed
             << std::setprecision(4) << epoch_loss << " | Speed: " << std::fixed << std::setprecision(2) << itps
             << " it/s"
             << " | Val Loss: " << std::fixed << std::setprecision(4) << val_loss << Vanilla(unix);

        // Print final line with newline
        std::cout << "\r" << std::string(120, ' ') << "\r" << line.str() << "\n" << std::flush;
    }
};