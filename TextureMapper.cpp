#include "TextureMapper.h"
#include "PatchMatch.h"

TextureMapper::TextureMapper() {
    runMain();
}

void TextureMapper::runMain() {
    cv::Mat source, cv::Mat target = init();
    align(source, target);
    reconstruct();
}

void TextureMapper::align(cv::Mat source, cv::Mat target) {
    patchSearch(source, target);
    vote();
}

/** source is a Mat that has all of the keyframes.
*
**/
void TextureMapper::patchSearch(cv::Mat source[], cv::Mat target[]) {
	
	// For each source pixel, output a 3-vector to the best match in
    // the target, with an error as the last channel.
	int sizes[3] = {source[0].width, source[0].height, /*frames*/ source.size()};
    cv::Mat out(3, sizes, CV_32FC(4));
	
	// Iterate over source frames, finding a match in the target where
    // the mask is high
	
	for (int t = 0; t < source.frames; t++) {
		// INITIALIZATION - uniform random assignment
        for (int y = 0; y < source.size().height; y++) {
            for (int x = 0; x < source.size().width; x++) {
                int dx = randomInt(patchSize, target.width-patchSize-1);
                int dy = randomInt(patchSize, target.height-patchSize-1);
                int dt = randomInt(0, target.frames-1);
				unsigned char* p = out.ptr(x, y, t) + 0;
                *p = dx;
				p = out.ptr(x, y, t) + 1;
				*p = dy;
				p = out.ptr(x, y, t) + 2;
				*p = dt;
				p = out.ptr(x, y, t) + 3;
				*p = distance(source, target, mask,
											x, y, t,
											dx, dy, dt,
											patchSize, HUGE_VAL);
            }
        }
	}
	
	bool forwardSearch = true;
	
	std::vector<cv::Mat> channels(4);
	cv::split(out, channels);
	cv::Mat dx = channels[0], dy = channels[1], dt = channels[2], error = channels[3];
	
}

float TextureMapper::distance(cv::Mat source, cv::Mat target, cv::Mat mask,
								int sx, int sy, int st,
								int tx, int ty, int tt,
								int patchSize, int floatThreshold) {
									
	// Do not use patches on boundaries
    if (tx < patchSize || tx >= target.width-patchSize ||
        ty < patchSize || ty >= target.height-patchSize) {
        return HUGE_VAL;
    }
	
	// Compute distance between patches
    // Average L2 distance in RGB space
    float dist = 0;
	
	int x1 = max(-patchSize, -sx, -tx);
    int x2 = min(patchSize, -sx+source.width-1, -tx+target.width-1);
    int y1 = max(-patchSize, -sy, -ty);
    int y2 = min(patchSize, -sy+source.height-1, -ty+target.height-1);

    for (int c = 0; c < target.channels; c++) {
        for (int y = y1; y <= y2; y++) {
            for (int x = x1; x <= x2; x++) {

                // Don't stray outside the mask
                if (mask.defined() && mask(tx+x, ty+y, tt, 0) < 1) return HUGE_VAL;

                float delta = source(sx+x, sy+y, st, c) - target(tx+x, ty+y, tt, c);
                dist += delta * delta;

                // Early termination
                if (dist > threshold) {return HUGE_VAL;}
            }
        }
    }

    return dist;
}

void TextureMapper::vote() {

}

void TextureMapper::reconstruct() {

}

cv::Mat TextureMapper::init() {
	
}