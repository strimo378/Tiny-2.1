/* tiny.c - TINY 2.1 as a plain serial console program.
 *
 * The same 367,000 parameter character transformer as TINY.EXE, with the
 * DOS screen, the PC speaker and the neural monitor taken out.  What is
 * left is what a device on the other end of a serial line would run: read
 * a line, answer it, print the answer, repeat.
 *
 * Portable C89.  No operating system beyond stdio, no heap, no threads,
 * no escape sequences, no terminal control, no timers in the answer path.
 * All memory is statically allocated and sized at compile time, so the
 * footprint is known before the program starts and is printed with -v.
 *
 *   cc -O2 -o tiny tiny.c -lm
 *   ./tiny                        reads TINY.BIN from the current directory
 *   ./tiny -f ../dos/TINY.BIN     weights from somewhere else
 *   ./tiny -q                     no banner and no prompt, for a pipe
 *   ./tiny -c 5                   creativity 0..9 (0 = always the surest)
 *
 * The network size is fixed at compile time (TINY_D, TINY_L, TINY_H,
 * TINY_HID below) and checked against the weight file header, so the
 * arrays are exactly as large as the model needs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef TINY_NO_CLOCK
#include <time.h>
#endif

/* ---- the network as it is shipped -------------------------------- */

#define TINY_VOCAB 97            /* 0 pad, 1 newline, 2.. ' '..'~'    */
#define TINY_CTX   128           /* question + answer, in letters     */

#ifndef TINY_D
#define TINY_D   96              /* width                             */
#endif
#ifndef TINY_L
#define TINY_L   4               /* layers                            */
#endif
#ifndef TINY_H
#define TINY_H   4               /* attention heads                   */
#endif
#ifndef TINY_HID
#define TINY_HID 256             /* middle width of the MLP           */
#endif

#define TINY_HD  (TINY_D / TINY_H)
#define TINY_MAXQ 72             /* longest question we accept         */

/* ---- weights ------------------------------------------------------
 * Every table is stored row by row: one float32 scale and then one
 * signed byte per weight.  A row of a table is dequantized as
 * byte * scale, which is what the 8-bit export in train2.py produces.
 */
typedef struct {
    const signed char *w;        /* rows * cols bytes                 */
    const float *scale;          /* one per row                       */
    int rows, cols;
} Table;

#define W(name, rows, cols) \
    static signed char name##_w[(rows) * (cols)]; \
    static float name##_s[rows]

W(emb, TINY_VOCAB, TINY_D);
W(pos, TINY_CTX, TINY_D);
W(wq, TINY_L *TINY_D, TINY_D);
W(wk, TINY_L *TINY_D, TINY_D);
W(wv, TINY_L *TINY_D, TINY_D);
W(wo, TINY_L *TINY_D, TINY_D);
W(f1, TINY_L *TINY_HID, TINY_D);
W(f2, TINY_L *TINY_D, TINY_HID);

static float g1[TINY_L][TINY_D];              /* normalisation gains    */
static float g2[TINY_L][TINY_D];
static float gf[TINY_D];

/* What earlier letters left behind: the key and value lists, kept as
 * bytes with one scale per list.  This is the only large working buffer
 * and the reason the whole thing fits in half a megabyte. */
static signed char kc[TINY_L][TINY_CTX][TINY_D];
static signed char vc[TINY_L][TINY_CTX][TINY_D];
static float ks[TINY_L][TINY_CTX];
static float vs[TINY_L][TINY_CTX];

/* Work buffers.  Static, not on the stack: an embedded stack is small. */
static float bx[TINY_D], bh[TINY_D], bq[TINY_D], bt[TINY_D], ba[TINY_D];
static float bm[TINY_HID], bs[TINY_CTX > TINY_VOCAB ? TINY_CTX : TINY_VOCAB];
static float score[TINY_VOCAB];

static const Table t_emb = { emb_w, emb_s, TINY_VOCAB, TINY_D };

/* one layer's table, as a window into the arrays above */
static Table layer(const signed char *w, const float *s, int l, int rows, int cols)
{
    Table t;
    t.w = w + (long)l * rows * cols;
    t.scale = s + (long)l * rows;
    t.rows = rows;
    t.cols = cols;
    return t;
}

#define L_WQ(l) layer(wq_w, wq_s, (l), TINY_D, TINY_D)
#define L_WK(l) layer(wk_w, wk_s, (l), TINY_D, TINY_D)
#define L_WV(l) layer(wv_w, wv_s, (l), TINY_D, TINY_D)
#define L_WO(l) layer(wo_w, wo_s, (l), TINY_D, TINY_D)
#define L_F1(l) layer(f1_w, f1_s, (l), TINY_HID, TINY_D)
#define L_F2(l) layer(f2_w, f2_s, (l), TINY_D, TINY_HID)

/* ---- reading the weights ----------------------------------------- */

static FILE *src;

static int src_open(const char *name)
{
    src = fopen(name, "rb");
    return src != NULL;
}

static int src_read(void *dst, unsigned long n)
{
    return fread(dst, 1, (size_t)n, src) == n;
}

static int src_at_end(void) { return fgetc(src) == EOF; }
static void src_close(void) { if (src) fclose(src); }

static void die(const char *msg)
{
    printf("tiny: %s\n", msg);
    exit(1);
}

/* little endian float32 and int16, whatever this machine is */
static float rd_float(void)
{
    unsigned char b[4];
    unsigned long bits;
    float f;
    if (!src_read(b, 4))
        die("the weight file is too short");
    bits = (unsigned long)b[0] | ((unsigned long)b[1] << 8) |
           ((unsigned long)b[2] << 16) | ((unsigned long)b[3] << 24);
    if (sizeof(float) == 4) {
        memcpy(&f, &bits, 4);                  /* the common case       */
        return f;
    } else {                                   /* no IEEE float32 here  */
        int sign = (int)(bits >> 31), exp = (int)((bits >> 23) & 0xFF);
        double frac = (double)(bits & 0x7FFFFFUL) / 8388608.0;
        double v = exp == 0 ? frac * 2.0 : (1.0 + frac) * pow(2.0, exp - 127);
        return (float)(sign ? -v : v);
    }
}

static int rd_int16(void)
{
    unsigned char b[2];
    if (!src_read(b, 2))
        die("the weight file is too short");
    return (int)b[0] | ((int)b[1] << 8);
}

static void rd_table(signed char *w, float *scale, int rows, int cols)
{
    int r;
    for (r = 0; r < rows; r++) {
        scale[r] = rd_float();
        if (!src_read(w + (long)r * cols, (unsigned long)cols))
            die("the weight file is too short");
    }
}

static void rd_gains(float *g)
{
    int i;
    for (i = 0; i < TINY_D; i++)
        g[i] = rd_float();
}

/* 'TNY2', D, L, H, HID as int16, then the tables in a fixed order. */
static void load_weights(const char *name)
{
    char magic[4];
    int l;

    if (!src_open(name))
        die("cannot open the weight file (TINY.BIN)");
    if (!src_read(magic, 4) || memcmp(magic, "TNY2", 4) != 0)
        die("not a version 2 weight file");
    if (rd_int16() != TINY_D || rd_int16() != TINY_L ||
        rd_int16() != TINY_H || rd_int16() != TINY_HID)
        die("the weight file has a different network size than this build\n"
            "       (rebuild with -DTINY_D=.. -DTINY_L=.. -DTINY_H=.. -DTINY_HID=..)");

    rd_table(emb_w, emb_s, TINY_VOCAB, TINY_D);
    rd_table(pos_w, pos_s, TINY_CTX, TINY_D);
    for (l = 0; l < TINY_L; l++) {
        rd_gains(g1[l]);
        rd_table(wq_w + (long)l * TINY_D * TINY_D, wq_s + l * TINY_D, TINY_D, TINY_D);
        rd_table(wk_w + (long)l * TINY_D * TINY_D, wk_s + l * TINY_D, TINY_D, TINY_D);
        rd_table(wv_w + (long)l * TINY_D * TINY_D, wv_s + l * TINY_D, TINY_D, TINY_D);
        rd_table(wo_w + (long)l * TINY_D * TINY_D, wo_s + l * TINY_D, TINY_D, TINY_D);
        rd_gains(g2[l]);
        rd_table(f1_w + (long)l * TINY_HID * TINY_D, f1_s + l * TINY_HID, TINY_HID, TINY_D);
        rd_table(f2_w + (long)l * TINY_D * TINY_HID, f2_s + l * TINY_D, TINY_D, TINY_HID);
    }
    rd_gains(gf);
    if (!src_at_end())
        die("the weight file is longer than this network size");
    src_close();
}

/* ---- the network ------------------------------------------------- */

/* y = table * x */
static void matvec(const Table *t, const float *x, float *y)
{
    const signed char *w = t->w;
    int r, c, cols = t->cols, rows = t->rows;
    float s;
    for (r = 0; r < rows; r++) {
        s = 0.0f;
        for (c = 0; c < cols; c++)
            s += (float)w[c] * x[c];
        y[r] = s * t->scale[r];
        w += cols;
    }
}

/* y = x brought to a standard size, times the gains g */
static void norm(const float *x, const float *g, float *y)
{
    int i;
    float s = 0.0f;
    for (i = 0; i < TINY_D; i++)
        s += x[i] * x[i];
    s = (float)(1.0 / sqrt((double)s / TINY_D + 1e-5));
    for (i = 0; i < TINY_D; i++)
        y[i] = x[i] * s * g[i];
}

/* e to the power x, 0 for very negative x (no underflow warnings) */
static float safe_exp(float x)
{
    return x < -60.0f ? 0.0f : (float)exp((double)x);
}

/* keep D numbers as bytes, returning the scale that brings them back */
static float store(signed char *dst, const float *src_v)
{
    int i;
    float big = 0.0f, v, k;
    for (i = 0; i < TINY_D; i++) {
        v = src_v[i] < 0.0f ? -src_v[i] : src_v[i];
        if (v > big) big = v;
    }
    if (big < 1e-20f)
        big = 1e-20f;
    k = 127.0f / big;
    for (i = 0; i < TINY_D; i++) {
        v = src_v[i] * k;
        dst[i] = (signed char)(v < 0.0f ? v - 0.5f : v + 0.5f);
    }
    return big / 127.0f;
}

/* Feed letter `tok` at position `p`; fills score[] with how well each
 * letter fits as the next one. */
static void step_once(int tok, int p)
{
    const float att_div = (float)sqrt((double)TINY_HD);
    const signed char *e = emb_w + (long)tok * TINY_D;
    const signed char *q = pos_w + (long)p * TINY_D;
    const signed char *kp, *vp;
    float es = emb_s[tok], ps = pos_s[p], s, big, sum;
    int i, j, l, h;
    Table t;

    for (i = 0; i < TINY_D; i++)
        bx[i] = (float)e[i] * es + (float)q[i] * ps;

    for (l = 0; l < TINY_L; l++) {
        /* look back at the earlier letters */
        norm(bx, g1[l], bh);
        t = L_WK(l); matvec(&t, bh, bt);
        ks[l][p] = store(kc[l][p], bt);
        t = L_WV(l); matvec(&t, bh, bt);
        vs[l][p] = store(vc[l][p], bt);
        t = L_WQ(l); matvec(&t, bh, bq);
        for (h = 0; h < TINY_H; h++) {
            big = -1e30f;
            for (j = 0; j <= p; j++) {
                kp = kc[l][j] + h * TINY_HD;
                s = 0.0f;
                for (i = 0; i < TINY_HD; i++)
                    s += bq[h * TINY_HD + i] * (float)kp[i];
                s = s * ks[l][j] / att_div;
                bs[j] = s;
                if (s > big) big = s;
            }
            sum = 0.0f;
            for (j = 0; j <= p; j++) {
                bs[j] = safe_exp(bs[j] - big);
                sum += bs[j];
            }
            for (i = 0; i < TINY_HD; i++)
                ba[h * TINY_HD + i] = 0.0f;
            for (j = 0; j <= p; j++) {
                if (bs[j] == 0.0f)
                    continue;
                vp = vc[l][j] + h * TINY_HD;
                s = bs[j] * vs[l][j] / sum;
                for (i = 0; i < TINY_HD; i++)
                    ba[h * TINY_HD + i] += s * (float)vp[i];
            }
        }
        t = L_WO(l); matvec(&t, ba, bt);
        for (i = 0; i < TINY_D; i++)
            bx[i] += bt[i];

        /* think about it */
        norm(bx, g2[l], bh);
        t = L_F1(l); matvec(&t, bh, bm);
        for (i = 0; i < TINY_HID; i++)
            if (bm[i] < 0.0f) bm[i] = 0.0f;
        t = L_F2(l); matvec(&t, bm, bt);
        for (i = 0; i < TINY_D; i++)
            bx[i] += bt[i];
    }
    norm(bx, gf, bh);
    matvec(&t_emb, bh, score);
}

/* ---- choosing letters -------------------------------------------- */

static int creativity = 0;                  /* 0 = always the surest      */
static unsigned long rng_state = 2463534242UL;

static unsigned long rnd(void)              /* xorshift32, no library rand */
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* With creativity 0 always the best letter.  Otherwise one of the letters
 * that fit nearly as well, so the same question gets differently worded
 * answers. */
static int pick(void)
{
    int i, best = 0;
    float t, sum = 0.0f, r;

    for (i = 1; i < TINY_VOCAB; i++)
        if (score[i] > score[best]) best = i;
    if (creativity <= 0)
        return best;
    t = (float)creativity / 5.0f;
    for (i = 0; i < TINY_VOCAB; i++) {
        bs[i] = safe_exp((score[i] - score[best]) / t);
        if (bs[i] < 0.15f) bs[i] = 0.0f;    /* never pick unlikely letters */
        sum += bs[i];
    }
    r = sum * (float)(rnd() % 10000UL) / 10000.0f;
    for (i = 0; i < TINY_VOCAB; i++) {
        r -= bs[i];
        if (r < 0.0f && bs[i] > 0.0f) return i;
    }
    return best;
}

/* Answer `q` into out[], which needs TINY_CTX + 1 bytes.  Returns the
 * number of letters. */
static int answer(const char *q, char *out)
{
    int p = 0, tok, letters = 0;

    for (; q[p]; p++)
        step_once(q[p] - 30, p);
    step_once(1, p++);                      /* end of the question         */
    for (; p < TINY_CTX; p++) {
        tok = pick();
        if (tok <= 1)                       /* end of the answer          */
            break;
        out[letters++] = (char)(tok + 30);
        step_once(tok, p);
    }
    out[letters] = '\0';
    return letters;
}

/* ---- the serial console ------------------------------------------ */

/* Printable letters only, no leading or trailing spaces, CAPS LOCK text
 * turned into lower case (the network knows it better).  CR and LF both
 * end a line, so a terminal that sends CRLF works as well. */
static int clean(const char *line, char *q)
{
    int i, n = 0, upper = 0, lower = 0;
    for (i = 0; line[i] && line[i] != '\n' && line[i] != '\r'; i++) {
        unsigned char c = (unsigned char)line[i];
        if (c < 32 || c > 126)
            continue;
        if (c >= 'A' && c <= 'Z') upper++;
        if (c >= 'a' && c <= 'z') lower++;
        if (c == ' ' && n == 0)
            continue;                       /* no leading spaces          */
        if (n < TINY_MAXQ)
            q[n++] = (char)c;
    }
    while (n > 0 && q[n - 1] == ' ')
        n--;
    q[n] = '\0';
    if (upper > lower)
        for (i = 0; i < n; i++)
            if (q[i] >= 'A' && q[i] <= 'Z')
                q[i] = (char)(q[i] + 32);
    return n;
}

static int same_word(const char *a, const char *b)
{
    int i;
    for (i = 0; a[i] && b[i]; i++)
        if ((a[i] | 32) != b[i]) return 0;
    return a[i] == '\0' && b[i] == '\0';
}

/* one line from the wire; returns 0 at end of input */
static int read_line(char *line, int max)
{
    int n = 0, c;
    for (;;) {
        c = getchar();
        if (c == EOF)
            return n > 0 ? (line[n] = '\0', 1) : 0;
        if (c == '\n' || c == '\r') {
            line[n] = '\0';
            return 1;
        }
        if (n < max - 1)
            line[n++] = (char)c;
    }
}

static unsigned long footprint(void)
{
    return (unsigned long)(sizeof emb_w + sizeof emb_s + sizeof pos_w + sizeof pos_s +
                           sizeof wq_w + sizeof wq_s + sizeof wk_w + sizeof wk_s +
                           sizeof wv_w + sizeof wv_s + sizeof wo_w + sizeof wo_s +
                           sizeof f1_w + sizeof f1_s + sizeof f2_w + sizeof f2_s +
                           sizeof g1 + sizeof g2 + sizeof gf +
                           sizeof kc + sizeof vc + sizeof ks + sizeof vs +
                           sizeof bx + sizeof bh + sizeof bq + sizeof bt + sizeof ba +
                           sizeof bm + sizeof bs + sizeof score);
}

static void usage(void)
{
    printf("TINY 2.1 - a question answering neural network, %d parameters\n",
           TINY_VOCAB * TINY_D + TINY_CTX * TINY_D +
           TINY_L * (4 * TINY_D * TINY_D + 2 * TINY_D * TINY_HID + 2 * TINY_D) + TINY_D);
    printf("usage: tiny [-f FILE] [-c 0..9] [-s SEED] [-q] [-v] [-h]\n"
           "  -f FILE   weight file (default TINY.BIN)\n"
           "  -c 0..9   creativity: 0 always the surest answer, 9 the most varied\n"
           "  -s SEED   seed for the random choices (default fixed, so repeatable)\n"
           "  -q        no banner, no prompt, no timing: just the answers\n"
           "  -v        print the memory footprint and the network size\n"
           "  -h        this text\n"
           "Type a question and press Enter. QUIT or end of input stops.\n");
}

int main(int argc, char **argv)
{
    static char line[256], q[TINY_MAXQ + 1], out[TINY_CTX + 1];
    const char *file = "TINY.BIN";
    int quiet = 0, verbose = 0, i, letters;
#ifndef TINY_NO_CLOCK
    clock_t t0;
    long ms;
#endif

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
            file = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc)
            creativity = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            rng_state = (unsigned long)strtoul(argv[++i], NULL, 10) | 1UL;
        else if (strcmp(argv[i], "-q") == 0)
            quiet = 1;
        else if (strcmp(argv[i], "-v") == 0)
            verbose = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else {
            printf("tiny: unknown option %s (-h for help)\n", argv[i]);
            return 2;
        }
    }
    if (creativity < 0) creativity = 0;
    if (creativity > 9) creativity = 9;

    load_weights(file);

    if (verbose) {
        printf("network   %d wide, %d layers, %d heads, %d middle\n",
               TINY_D, TINY_L, TINY_H, TINY_HID);
        printf("weights   %lu bytes\n",
               (unsigned long)(sizeof emb_w + sizeof emb_s + sizeof pos_w + sizeof pos_s +
                               sizeof wq_w + sizeof wq_s + sizeof wk_w + sizeof wk_s +
                               sizeof wv_w + sizeof wv_s + sizeof wo_w + sizeof wo_s +
                               sizeof f1_w + sizeof f1_s + sizeof f2_w + sizeof f2_s +
                               sizeof g1 + sizeof g2 + sizeof gf));
        printf("letter memory %lu bytes, work buffers %lu bytes\n",
               (unsigned long)(sizeof kc + sizeof vc + sizeof ks + sizeof vs),
               (unsigned long)(sizeof bx + sizeof bh + sizeof bq + sizeof bt +
                               sizeof ba + sizeof bm + sizeof bs + sizeof score));
        printf("total     %lu bytes of RAM, no heap\n", footprint());
    }
    if (!quiet) {
        printf("TINY 2.1 ready. Ask about countries, animals, space, the body,\n");
        printf("nature, science, food, sums, numbers, spelling, words, shapes,\n");
        printf("history, music or computers. QUIT stops.\n");
    }

    for (;;) {
        if (!quiet) {
            printf("\n> ");
            fflush(stdout);
        }
        if (!read_line(line, (int)sizeof line))
            break;
        if (clean(line, q) == 0)
            continue;
        if (same_word(q, "quit") || same_word(q, "exit"))
            break;
#ifndef TINY_NO_CLOCK
        t0 = clock();
#endif
        letters = answer(q, out);
        if (letters == 0)
            strcpy(out, ".");
        printf("%s\n", out);
#ifndef TINY_NO_CLOCK
        ms = (long)((double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC);
        if (!quiet)
            printf("[%d letters, %ld ms]\n", letters, ms);
#endif
        fflush(stdout);
    }
    if (!quiet)
        printf("\nbye\n");
    return 0;
}
