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

void TextureMapper::patchSearch(cv::Mat source, cv::Mat target) {
	for (int t = 0; t < source.frames; t++) {
		// INITIALIZATION - uniform random assignment
        for (int y = 0; y < source.size().height; y++) {
            for (int x = 0; x < source.size().width; x++) {
                int dx = randomInt(patchSize, target.width-patchSize-1);
                int dy = randomInt(patchSize, target.height-patchSize-1);
                int dt = randomInt(0, target.frames-1);
                out(x, y, t, 0) = dx;
                out(x, y, t, 1) = dy;
                out(x, y, t, 2) = dt;
                out(x, y, t, 3) = distance(source, target, mask,
                                           x, y, t,
                                           dx, dy, dt,
                                           patchSize, HUGE_VAL);
            }
        }
	}
	
	bool forwardSearch = true;
	cv::Mat dx = out.channel(0), dy = out.channel(1), dt = out.channel(2), error = out.channel(3);
	
}

void TextureMapper::vote() {

}

void TextureMapper::reconstruct() {

}