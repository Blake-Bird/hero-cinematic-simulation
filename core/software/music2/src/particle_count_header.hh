// Native HDF5 particle-count attributes. Full counts match GADGET-4's
// non-GADGET2_HEADER reader; legacy split totals remain available explicitly.
#pragma once
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "plugins/HDF_IO.hh"

namespace music_particle_counts {
inline void write(const std::string &filename,
                  const std::vector<unsigned long long> &local,
                  const std::vector<long long> &totals, bool modern) {
    if(local.size()!=6 || totals.size()!=local.size())
        throw std::runtime_error("Native particle count header requires six species");
    std::vector<unsigned long long> full(totals.size());
    std::vector<unsigned int> low(totals.size()),high(totals.size()),classic_local(local.size());
    unsigned long long global=0;
    for(size_t i=0;i<totals.size();++i) {
        if(totals[i]<0 || local[i]>static_cast<unsigned long long>(totals[i]))
            throw std::runtime_error("Invalid exact native particle counts");
        full[i]=static_cast<unsigned long long>(totals[i]);
        if(full[i]>static_cast<unsigned long long>(std::numeric_limits<long long>::max())-global)
            throw std::runtime_error("Native total particle count exceeds signed 64-bit storage");
        global+=full[i];
        low[i]=static_cast<unsigned int>(full[i]);
        high[i]=modern?0:static_cast<unsigned int>(full[i]>>32);
        if(!modern && local[i]>std::numeric_limits<unsigned int>::max())
            throw std::runtime_error("Legacy per-file particle count would truncate; use full 64-bit count encoding");
        classic_local[i]=static_cast<unsigned int>(local[i]);
    }
    if(modern) {
        HDFWriteGroupAttribute(filename,"Header","NumPart_ThisFile",local);
        HDFWriteGroupAttribute(filename,"Header","NumPart_Total",full);
    } else {
        HDFWriteGroupAttribute(filename,"Header","NumPart_ThisFile",classic_local);
        HDFWriteGroupAttribute(filename,"Header","NumPart_Total",low);
    }
    HDFWriteGroupAttribute(filename,"Header","NumPart_Total_HighWord",high);
    HDFWriteGroupAttribute(filename,"Header","Music_particle_count_schema",modern?1:0);
}
}
