#include "terminal.h"

#if defined(_WIN64) || defined(_WIN32)
#include <Windows.h>
#endif

bool get_unix()
{
    bool unix_style;

#if defined(_WIN64) || defined(_WIN32)
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(h, &mode);
    unix_style = SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
    unix_style = true;
#endif

    return unix_style;
}