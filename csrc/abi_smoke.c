#include <stdio.h>
#include "libwhale.h"
int main(void) {
    WhaleHandle *h = whale_create();
    if (h == 0) {
        return 1;
    }
    if (whale_set_startpos(h) != 0) {
        whale_destroy(h);
        return 1;
    }
    char best[16];
    int rc = whale_go_depth(h, 8, best, sizeof(best));
    if (rc == 0) {
        printf("%s\n", best);
    }
    whale_destroy(h);
    return rc != 0;
}
