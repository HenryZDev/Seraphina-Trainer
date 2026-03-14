#pragma once

#define Vanilla(unix) (unix ? "\033[0m" : "")
#define Bold(unix) (unix ? "\033[1m" : "")
#define Halve(unix) (unix ? "\033[2m" : "")
#define Italic(unix) (unix ? "\033[3m" : "")
#define Underline(unix) (unix ? "\033[4m" : "")
#define Blink(unix) (unix ? "\033[5m" : "")
#define Flash(unix) (unix ? "\033[6m" : "")
#define Invert(unix) (unix ? "\033[7m" : "")
#define Fade(unix) (unix ? "\033[8m" : "")
#define Strikethrough(unix) (unix ? "\033[9m" : "")

#define Black(unix) (unix ? "\033[30m" : "")
#define Red(unix) (unix ? "\033[31m" : "")
#define Green(unix) (unix ? "\033[32m" : "")
#define Yellow(unix) (unix ? "\033[33m" : "")
#define Blue(unix) (unix ? "\033[34m" : "")
#define Purple(unix) (unix ? "\033[35m" : "")
#define Cyan(unix) (unix ? "\033[36m" : "")
#define White(unix) (unix ? "\033[37m" : "")

#define UnderlineOn(unix) (unix ? "\033[38m" : "")
#define UnderlineOff(unix) (unix ? "\033[39m" : "")

#define BlackBG(unix) (unix ? "\033[40m" : "")
#define RedBG(unix) (unix ? "\033[41m" : "")
#define GreenBG(unix) (unix ? "\033[42m" : "")
#define BrownBG(unix) (unix ? "\033[43m" : "")
#define BlueBG(unix) (unix ? "\033[44m" : "")
#define MagentaBG(unix) (unix ? "\033[45m" : "")
#define PeacockBlueBG(unix) (unix ? "\033[46m" : "")
#define WhiteBG(unix) (unix ? "\033[47m" : "")

bool get_unix();