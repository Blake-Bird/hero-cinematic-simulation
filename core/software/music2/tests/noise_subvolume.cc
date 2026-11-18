// Compare actual periodic subvolume noise to independently generated raw GSL
// cubes. Exercise repeated periodic cube visits at several thread counts.
#include "plugins/random_music_wnoise_generator.hh"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main()
{
    using noise = music_wnoise_generator<real_t>;
    size_t checked = 0;
    for (int side : {32, 64, 128}) {
        const int cube_size = 32, cubes = side / cube_size;
        const long seed = 10421;
        std::vector<double> expected(size_t(side) * side * side);
        gsl_rng *generator = gsl_rng_alloc(gsl_rng_mt19937);
        for (int a=0; a<cubes; ++a)
            for (int b=0; b<cubes; ++b)
                for (int c=0; c<cubes; ++c) {
                    gsl_rng_set(generator, seed + (a*cubes+b)*cubes+c);
                    for (int i=0; i<cube_size; ++i)
                        for (int j=0; j<cube_size; ++j)
                            for (int k=0; k<cube_size; ++k)
                                expected[(size_t(a*cube_size+i)*side+b*cube_size+j)*side+c*cube_size+k]
                                    = static_cast<real_t>(gsl_ran_ugaussian_ratio_method(generator));
                }
        gsl_rng_free(generator);
        for (int threads : {1, 2, 8}) {
            omp_set_num_threads(threads);
            for (int repeat=0; repeat<8; ++repeat) {
                int origin[3] = {-4+repeat, side-7, 3};
                int extent[3] = {side+8, side+4, side+2};
                noise actual(side, cube_size, seed, origin, extent);
                for (int i=0; i<extent[0]; ++i)
                    for (int j=0; j<extent[1]; ++j)
                        for (int k=0; k<extent[2]; ++k) {
                            const int x=(origin[0]+i+side)%side;
                            const int y=(origin[1]+j+side)%side;
                            const int z=(origin[2]+k+side)%side;
                            // Use normalized coordinates: this test isolates cube
                            // scheduling from the separate accessor wrap contract.
                            if (actual(x,y,z) != expected[(size_t(x)*side+y)*side+z]) {
                                std::cerr << "Noise mismatch side=" << side << " threads=" << threads << '\n';
                                return 1;
                            }
                            ++checked;
                        }
            }
        }
    }
    std::cout << "PASS raw noise values=" << checked << " threads=1,2,8 sides=32,64,128\n";
    return 0;
}
