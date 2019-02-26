#include "TextureMapper.h"

TextureMapper::runMain() {
    init();
    while(!converged) {
        align();
        reconstruct();
    }
}
