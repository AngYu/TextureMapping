#include "TextureMapper.h"
#include "PatchMatch.h"

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
    PatchMatch patchMatcher;
    patchMatcher = PatchMatch();
}

void TextureMapper::vote() {

}

void TextureMapper::reconstruct() {

}