// Exact synchronized snapshot boundaries on GADGET's integer timeline.
// Steps remain dyadic and are never enlarged. The usual alignment checks still
// apply; all active physics must stop before the next output boundary.
#ifndef RESEARCH_EXACT_OUTPUT_TIMELINE_H
#define RESEARCH_EXACT_OUTPUT_TIMELINE_H

template <typename Integer>
inline Integer research_output_step_limit(Integer step, Integer current,
                                          Integer next_output, Integer timebase)
{
  if(next_output > current && next_output <= timebase && step > next_output - current)
    {
      Integer limit = next_output - current;
      Integer dyadic = 1;
      while(dyadic <= limit / 2)
        dyadic *= 2;
      return dyadic;
    }
  return step;
}
#endif
