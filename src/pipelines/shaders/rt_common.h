#ifndef RT_COMMON_H_INCLUDED
#define RT_COMMON_H_INCLUDED

struct RTPayload {
#ifdef OUTPUT_COLOR
    vec3 color;
#endif

#ifdef OUTPUT_DEPTH
    float depth;
#endif
};

#endif
