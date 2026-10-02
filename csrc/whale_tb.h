#ifndef WHALE_TB_H
#define WHALE_TB_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int whale_tb_init(const char *path);
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
    int *wdl_out);
void whale_tb_free(void);
#ifdef __cplusplus
}
#endif
#endif
