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

// FIXME use r / u
vec3 computeRayDir(uvec2 pixel_coords, mat4 view_inv, mat4 proj_inv)
{
    const vec2 raster = vec2(pixel_coords) + vec2(0.5);
    const vec2 norm_raster = raster / vec2(gl_LaunchSizeEXT.xy);
    vec2 screen = norm_raster * 2.0 - 1.0;
    
    vec4 cam_pos = proj_inv * vec4(screen.x, screen.y, 1, 1);
    vec4 d = view_inv * vec4(normalize(cam_pos.xyz), 0);
    
    return d.xyz;
}

#endif
