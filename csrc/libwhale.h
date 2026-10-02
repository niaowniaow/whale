#ifndef LIBWHALE_H
#define LIBWHALE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct WhaleHandle WhaleHandle;
const char *whale_version(void);
WhaleHandle *whale_create(void);
void whale_destroy(WhaleHandle *h);
int whale_set_startpos(WhaleHandle *h);
int whale_set_fen(WhaleHandle *h, const char *fen);
int whale_push_uci(WhaleHandle *h, const char *uci);
int whale_go_depth(WhaleHandle *h, uint8_t depth, char *out, size_t out_cap);
int whale_set_syzygy_path(const char *path);
int whale_tb_probe_wdl(WhaleHandle *h);
#ifdef __cplusplus
}
#endif
#endif
