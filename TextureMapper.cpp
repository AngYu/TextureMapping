#include "TextureMapper.h"

void TextureMapper::runMain() {
    init();
    while(!converged) {
        align();
        reconstruct();
    }
}

void TextureMapper::align() {
    patchSearch();
    vote();
}

void TextureMapper