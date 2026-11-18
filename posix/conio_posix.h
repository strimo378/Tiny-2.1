/* conio_posix.h - the handful of Turbo C screen, keyboard, sound and timer
   calls that TINY.C uses, on a POSIX terminal (Cygwin, Linux, macOS).

   An 80x25 text screen with one attribute byte per cell is kept in memory,
   exactly like the DOS text screen, and every change is written to the
   terminal as ANSI escapes.  That way window(), scrolling, wherex/wherey
   and gettext() behave as they do under DOS and the TINY.C interface code
   needs no changes.

   DOS colour attributes become real colours: with the neon palette on, the
   same RGB values the DOS version writes into the VGA DAC are sent to the
   terminal as 24-bit colours; otherwise the standard 16 ANSI colours are
   used.

   Only included when TINY_POSIX is defined; the DOS build still uses the
   real <conio.h> and <dos.h>.  Single translation unit, so everything is
   static. */

#ifndef CONIO_POSIX_H
#define CONIO_POSIX_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

/* ---- text modes, as Turbo C names them --------------------------- */
#define BW40  0
#define C40   1
#define BW80  2
#define C80   3
#define MONO  7

struct text_info {
    unsigned char winleft, wintop, winright, winbottom;
    unsigned char attribute, normattr;
    unsigned char currmode;
    unsigned char screenheight, screenwidth;
    unsigned char curx, cury;
};

/* ---- a millisecond wall clock ------------------------------------ */
/* Cygwin's clock() only moves in 10-15 ms steps, far too coarse to time
   an answer on a modern machine. */
static long pc_millis(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long)(tv.tv_sec % 100000L) * 1000L + tv.tv_usec / 1000L;
}

/* ---- the clock, as dos.h gives it -------------------------------- */
struct time {
    unsigned char ti_min, ti_hour, ti_hund, ti_sec;
};

static void gettime(struct time *t)
{
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);
    t->ti_hour = (unsigned char)(lt ? lt->tm_hour : 0);
    t->ti_min = (unsigned char)(lt ? lt->tm_min : 0);
    t->ti_sec = (unsigned char)(lt ? lt->tm_sec : 0);
    t->ti_hund = 0;
}

#define PC_W 80
#define PC_H 25

static unsigned char pc_ch[PC_H][PC_W];          /* what is on the screen */
static unsigned char pc_at[PC_H][PC_W];
static int pc_wl = 1, pc_wt = 1, pc_wr = PC_W, pc_wb = PC_H;   /* window, 1-based */
static int pc_x = 1, pc_y = 1;                   /* cursor inside the window */
static int pc_attr = 7;
static int pc_neon;                              /* neon palette wanted */
static int pc_out_x = -1, pc_out_y = -1;         /* where the terminal cursor is */
static int pc_out_attr = -1;
static int pc_raw;                               /* raw mode is on */
static struct termios pc_saved_tty;

/* The colours the DOS version programs into the VGA DAC (0..63 each). */
static const unsigned char pc_neon_rgb[16][3] = {
    {  1,  1,  7 }, {  5,  7, 22 }, {  0, 26, 18 }, {  0, 20, 30 },
    { 40,  4, 12 }, { 28,  6, 40 }, { 42, 26,  4 }, { 40, 45, 54 },
    { 13, 16, 28 }, { 18, 34, 63 }, { 18, 63, 42 }, { 10, 57, 63 },
    { 63, 22, 34 }, { 63, 18, 58 }, { 63, 54, 14 }, { 60, 62, 63 }
};

/* ---- code page 437 ----------------------------------------------- */
/* The DOS version draws its frames, arrows and blocks with code page 437
   characters.  They are sent to the terminal as UTF-8; set TINY_ASCII=1
   for a terminal that cannot show them. */
static const unsigned short pc_cp437[256] = {
    0x0020, 0x263A, 0x263B, 0x2665, 0x2666, 0x2663, 0x2660, 0x2022,
    0x25D8, 0x25CB, 0x25D9, 0x2642, 0x2640, 0x266A, 0x266B, 0x263C,
    0x25BA, 0x25C4, 0x2195, 0x203C, 0x00B6, 0x00A7, 0x25AC, 0x21A8,
    0x2191, 0x2193, 0x2192, 0x2190, 0x221F, 0x2194, 0x25B2, 0x25BC,
    0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
    0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F,
    0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F,
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
    0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
    0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x2302,
    0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7,
    0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
    0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9,
    0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
    0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA,
    0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
    0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
    0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
    0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x00B5, 0x03C4,
    0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
    0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248,
    0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0
};

static int pc_ascii_only;                        /* TINY_ASCII=1 */

/* a plain ASCII stand-in for the characters TINY draws with */
static int pc_ascii(int ch)
{
    switch (ch) {
    case 0xB3: case 0xBA: case 0xB5: case 0xC6: case 0xB4: case 0xC3:
        return '|';
    case 0xC4: case 0xCD:
        return '-';
    case 0xDA: case 0xBF: case 0xC0: case 0xD9: case 0xC9: case 0xBB:
    case 0xC8: case 0xBC: case 0xC5: case 0xCE: case 0xD1: case 0xCF:
    case 0xC2: case 0xC1:
        return '+';
    case 0xB0: case 0xB1: case 0xB2: case 0xDB: case 0xDC: case 0xDD:
    case 0xDE: case 0xDF: case 0xFE:
        return '#';
    case 0x10: case 0xAF: case 0x1A:
        return '>';
    case 0x11: case 0xAE: case 0x1B:
        return '<';
    case 0x18: case 0x1E:
        return '^';
    case 0x19: case 0x1F:
        return 'v';
    case 0xFA: case 0xF9: case 0x07:
        return '.';
    }
    return ch >= 32 && ch <= 126 ? ch : ' ';
}

/* ---- writing to the terminal ------------------------------------- */

static void pc_emit(const char *s)
{
    fputs(s, stdout);
}

static void pc_putglyph(int ch)
{
    unsigned u;
    if (pc_ascii_only) {
        putchar(pc_ascii(ch));
        return;
    }
    u = pc_cp437[ch & 0xFF];
    if (u < 0x80)
        putchar((int)u);
    else if (u < 0x800) {
        putchar((int)(0xC0 | (u >> 6)));
        putchar((int)(0x80 | (u & 0x3F)));
    } else {
        putchar((int)(0xE0 | (u >> 12)));
        putchar((int)(0x80 | ((u >> 6) & 0x3F)));
        putchar((int)(0x80 | (u & 0x3F)));
    }
}

static void pc_goto_out(int x, int y)            /* absolute, 1-based */
{
    char buf[24];
    if (x == pc_out_x && y == pc_out_y)
        return;
    sprintf(buf, "\033[%d;%dH", y, x);
    pc_emit(buf);
    pc_out_x = x;
    pc_out_y = y;
}

static void pc_set_attr_out(int a)
{
    char buf[64];
    int fg = a & 15, bg = (a >> 4) & 15;
    if (a == pc_out_attr)
        return;
    if (pc_neon) {
        const unsigned char *f = pc_neon_rgb[fg], *b = pc_neon_rgb[bg];
        sprintf(buf, "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm",
                f[0] * 255 / 63, f[1] * 255 / 63, f[2] * 255 / 63,
                b[0] * 255 / 63, b[1] * 255 / 63, b[2] * 255 / 63);
    } else {
        static const int ansi[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };   /* DOS -> ANSI */
        sprintf(buf, "\033[%d;%dm",
                (fg & 8 ? 90 : 30) + ansi[fg & 7], (bg & 8 ? 100 : 40) + ansi[bg & 7]);
    }
    pc_emit(buf);
    pc_out_attr = a;
}

/* put one cell on the screen and on the terminal */
static void pc_cell(int x, int y, int ch, int attr)
{
    if (x < 1 || x > PC_W || y < 1 || y > PC_H)
        return;
    pc_ch[y - 1][x - 1] = (unsigned char)ch;
    pc_at[y - 1][x - 1] = (unsigned char)attr;
    pc_goto_out(x, y);
    pc_set_attr_out(attr);
    pc_putglyph(ch);
    if (++pc_out_x > PC_W)                       /* the terminal may or may not
                                                    wrap; never rely on it */
        pc_out_x = -1;
}

static void pc_repaint_row(int y, int x1, int x2)
{
    int x;
    for (x = x1; x <= x2; x++)
        pc_cell(x, y, pc_ch[y - 1][x - 1], pc_at[y - 1][x - 1]);
}

/* ---- the Turbo C calls ------------------------------------------- */

static void textattr(int a)
{
    pc_attr = a & 0xFF;
}

static void window(int left, int top, int right, int bottom)
{
    pc_wl = left; pc_wt = top; pc_wr = right; pc_wb = bottom;
    pc_x = pc_y = 1;
}

static void gotoxy(int x, int y)
{
    pc_x = x;
    pc_y = y;
}

static int wherex(void) { return pc_x; }
static int wherey(void) { return pc_y; }

static void pc_scroll(void)                      /* window up by one line */
{
    int y, x;
    for (y = pc_wt; y < pc_wb; y++)
        for (x = pc_wl; x <= pc_wr; x++) {
            pc_ch[y - 1][x - 1] = pc_ch[y][x - 1];
            pc_at[y - 1][x - 1] = pc_at[y][x - 1];
        }
    for (x = pc_wl; x <= pc_wr; x++) {
        pc_ch[pc_wb - 1][x - 1] = ' ';
        pc_at[pc_wb - 1][x - 1] = (unsigned char)pc_attr;
    }
    for (y = pc_wt; y <= pc_wb; y++)
        pc_repaint_row(y, pc_wl, pc_wr);
}

static void pc_newline(void)
{
    if (pc_wt + pc_y - 1 >= pc_wb)
        pc_scroll();
    else
        pc_y++;
}

static int putch(int c)
{
    if (c == '\n')
        pc_newline();                            /* line feed only, like DOS */
    else if (c == '\r')
        pc_x = 1;
    else if (c == '\b') {
        if (pc_x > 1) pc_x--;
    } else if (c == '\a')
        ;                                        /* no bell */
    else {
        pc_cell(pc_wl + pc_x - 1, pc_wt + pc_y - 1, c, pc_attr);
        if (pc_wl + pc_x - 1 >= pc_wr) {
            pc_x = 1;
            pc_newline();
        } else
            pc_x++;
    }
    return c;
}

static int cputs(const char *s)
{
    while (*s)
        putch((unsigned char)*s++);
    fflush(stdout);
    return 0;
}

static int cprintf(const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    int n;
    va_start(ap, fmt);
    n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    cputs(buf);
    return n;
}

static void clrscr(void)
{
    int y, x;
    for (y = pc_wt; y <= pc_wb; y++)
        for (x = pc_wl; x <= pc_wr; x++)
            pc_cell(x, y, ' ', pc_attr);
    pc_x = pc_y = 1;
    fflush(stdout);
}

/* char and attribute pairs, like the DOS text screen; absolute coordinates */
static void pc_gettext(int left, int top, int right, int bottom, void *dest)
{
    unsigned char *p = (unsigned char *)dest;
    int y, x;
    for (y = top; y <= bottom; y++)
        for (x = left; x <= right; x++) {
            *p++ = pc_ch[y - 1][x - 1];
            *p++ = pc_at[y - 1][x - 1];
        }
}
#define gettext pc_gettext                       /* libintl has its own gettext */

static void gettextinfo(struct text_info *ti)
{
    memset(ti, 0, sizeof(*ti));
    ti->winleft = (unsigned char)pc_wl;
    ti->wintop = (unsigned char)pc_wt;
    ti->winright = (unsigned char)pc_wr;
    ti->winbottom = (unsigned char)pc_wb;
    ti->attribute = ti->normattr = (unsigned char)pc_attr;
    ti->currmode = C80;
    ti->screenheight = PC_H;
    ti->screenwidth = PC_W;
    ti->curx = (unsigned char)pc_x;
    ti->cury = (unsigned char)pc_y;
}

static void textmode(int mode)                   /* only used to reset colours */
{
    (void)mode;
    pc_neon = 0;
    pc_out_attr = -1;
    pc_emit("\033[0m");
    fflush(stdout);
}

/* ---- timing and sound -------------------------------------------- */

static void delay(unsigned ms)
{
    struct timeval tv;
    fflush(stdout);
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000L;
    select(0, NULL, NULL, NULL, &tv);
}

/* No PC speaker on a terminal: the robot voice keeps its rhythm (the
   delays stay) but stays silent.  /QUIET skips the delays as well. */
static void sound(unsigned hz) { (void)hz; }
static void nosound(void) { }

/* ---- keyboard ---------------------------------------------------- */

static int pc_pending[8], pc_npending;           /* 0 + scan code pairs */

static void pc_cooked(void)
{
    if (pc_raw) {
        tcsetattr(0, TCSANOW, &pc_saved_tty);
        pc_raw = 0;
    }
    pc_emit("\033[?25h\033[0m");
    fflush(stdout);
}

static void pc_rawmode(void)
{
    struct termios t;
    if (pc_raw || tcgetattr(0, &pc_saved_tty) != 0)
        return;
    t = pc_saved_tty;
    t.c_lflag &= ~(unsigned)(ICANON | ECHO | ISIG);
    t.c_iflag &= ~(unsigned)(IXON | ICRNL);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &t) != 0)
        return;
    pc_raw = 1;
    atexit(pc_cooked);
}

static int pc_waitbyte(int ms)                   /* -1 if nothing arrived */
{
    unsigned char c;
    fd_set fds;
    struct timeval tv;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000L;
    if (select(1, &fds, NULL, NULL, ms < 0 ? NULL : &tv) <= 0)
        return -1;
    return read(0, &c, 1) == 1 ? c : -1;
}

static void pc_push(int scan)                    /* a key TINY.C reads as 0, scan */
{
    if (pc_npending < (int)(sizeof(pc_pending) / sizeof(pc_pending[0])))
        pc_pending[pc_npending++] = scan;
}

/* Turn an escape sequence into the DOS scan code TINY.C expects, or 27 for
   a lone Esc.  F1..F4 = 59..62, cursor up = 72. */
static int pc_escape(void)
{
    int a = pc_waitbyte(40), b, n;
    if (a < 0)
        return 27;                               /* plain Esc */
    if (a == 'O' || a == '[') {
        b = pc_waitbyte(40);
        if (b < 0)
            return 27;
        switch (b) {
        case 'A': pc_push(72); return 0;         /* up arrow */
        case 'P': case 'Q': case 'R': case 'S':  /* xterm F1..F4 (SS3) */
            pc_push(59 + (b - 'P'));
            return 0;
        case '[':                                /* linux console F1..F4 */
            n = pc_waitbyte(40);
            if (n >= 'A' && n <= 'D') { pc_push(59 + (n - 'A')); return 0; }
            return -1;
        default:
            if (b >= '0' && b <= '9') {          /* [11~ .. [14~, [1;2P ... */
                char num[8];
                int i = 0;
                num[i++] = (char)b;
                while (i < (int)sizeof(num) - 1) {
                    n = pc_waitbyte(40);
                    if (n < 0 || n == '~' || n == ';')
                        break;
                    if (n < '0' || n > '9') {    /* e.g. [1;5A */
                        if (n == 'A') { pc_push(72); return 0; }
                        break;
                    }
                    num[i++] = (char)n;
                }
                num[i] = 0;
                i = atoi(num);
                if (i >= 11 && i <= 14) { pc_push(59 + (i - 11)); return 0; }
            }
            return -1;                           /* something we do not use */
        }
    }
    return -1;
}

static int getch(void)
{
    int c;
    pc_rawmode();
    if (pc_npending > 0) {                       /* second half of a special key */
        c = pc_pending[0];
        memmove(pc_pending, pc_pending + 1, (size_t)(--pc_npending) * sizeof(int));
        return c;
    }
    fflush(stdout);
    for (;;) {
        c = pc_waitbyte(-1);
        if (c < 0)
            return 27;                           /* stdin gone: behave like Esc */
        if (c == 27) {
            c = pc_escape();
            if (c < 0)
                continue;                        /* a sequence TINY does not use */
            return c;
        }
        if (c == 127)
            return 8;                            /* backspace */
        if (c == '\n')
            return 13;
        if (c == 3)
            return 27;                           /* Ctrl-C quits like Esc */
        return c;
    }
}

static int kbhit(void)
{
    fd_set fds;
    struct timeval tv;
    if (pc_npending > 0)
        return 1;
    pc_rawmode();
    fflush(stdout);
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    tv.tv_sec = tv.tv_usec = 0;
    return select(1, &fds, NULL, NULL, &tv) > 0;
}

/* ---- what int86 did on DOS -------------------------------------- */

static void pc_cursor(int on)
{
    pc_emit(on ? "\033[?25h" : "\033[?25l");
    fflush(stdout);
}

static int pc_has_truecolor(void)
{
    const char *s = getenv("COLORTERM");
    if (s && (strstr(s, "truecolor") || strstr(s, "24bit")))
        return 1;
    s = getenv("TERM");
    return s && (strstr(s, "256color") || strstr(s, "direct"));
}

static void pc_neon_palette(void)
{
    pc_neon = 1;
    pc_out_attr = -1;
}

/* The terminal must be at least 80x25, like a DOS text screen. */
static int pc_screen_fits(void)
{
    struct winsize ws;
    if (ioctl(1, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0)
        return 1;                                /* cannot tell: try anyway */
    return ws.ws_col >= PC_W && ws.ws_row >= PC_H;
}

static void pc_begin(void)
{
    const char *s = getenv("TINY_ASCII");
    pc_ascii_only = s != NULL && *s != '0';
    pc_rawmode();
    pc_emit("\033[2J");                          /* clear, then draw our screen */
    pc_out_x = pc_out_y = -1;
    pc_out_attr = -1;
}

#endif /* CONIO_POSIX_H */
