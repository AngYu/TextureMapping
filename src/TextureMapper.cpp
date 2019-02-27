#include "TextureMapper.h"

TextureMapper::TextureMapper() {
    runMain();
}

void TextureMapper::runMain() {
    init();
    align();
    reconstruct();
}

void TextureMapper::align() {
    patchSearch();
    vote();
}

void TextureMapper::patchSearch() {

}

void TextureMapper::vote() {

}

void TextureMapper::reconstruct() {

}