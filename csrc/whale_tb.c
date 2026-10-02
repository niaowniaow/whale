#include "whale_tb.h"
#include <stddef.h>
int whale_tb_init(const char *path) {
    (void)path;
    return -2;
}
int whale_tb_probe_wdl_raw(
    uint64_t white,
    uint64_t black,
    uint64_t kings,
    uint64_t queens,
    uint64_t rooks,
    uint64_t bishops,
    uint64_t knights,
    uint64_t pawns,
    int stm_white,
    int castling,
    int ep_square,
    int halfmove,
    int *wdl_out) {
    (void)white;
    (void)black;
    (void)kings;
    (void)queens;
    (void)rooks;
    (void)bishops;
    (void)knights;
    (void)pawns;
    (void)stm_white;
    (void)castling;
    (void)ep_square;
    (void)halfmove;
    (void)wdl_out;
    return -2;
}
void whale_tb_free(void) {
}
