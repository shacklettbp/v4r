#include <cstdlib>
#include <iostream>
#include <EGL/egl.h>
#include <dlfcn.h>

using namespace std;

static __attribute__((constructor)) void nvidiaLinuxHeadlessHacksEntry() 
{
    // The NVIDIA icd entry point is libGLX_nvidia.so.
    // When X11 forwarding is enabled through ssh and DISPLAY is set,
    // vkEnumeratePhysicalDevices crashes with VK_INITIALIZATION_ERROR
    unsetenv("DISPLAY");

    // When DISPLAY is unset, libGLX_nvidia.so dispatches all calls
    // to libEGL.so. On systems using libGLVND (arch for example),
    // libEGL.so does not know what vendor is running, because there
    // have been no calls to egl and it can't cheat and look at
    // the X server like libGLX_nvidia.so. Calling an EGL function sets
    // libGLVND's state correctly, which avoids a segfault on exit
    // when libGLX_nvidia.so calls eglReleaseThread, which would otherwise
    // mistakenly end up calling the function in libEGL_mesa.so
    void *lib = dlopen("libEGL.so", RTLD_NOW | RTLD_NODELETE);
    if (!lib) return;

    using eglGetDisplayType = decltype(&eglGetDisplay);
    auto egl_get_display_ptr = (eglGetDisplayType)dlsym(lib, "eglGetDisplay");
    if (!egl_get_display_ptr) return;
    egl_get_display_ptr(EGL_DEFAULT_DISPLAY);

    using eglReleaseThreadType = decltype(&eglReleaseThread);
    auto egl_release_thread_ptr = (eglReleaseThreadType)dlsym(lib, "eglReleaseThread");
    if (!egl_release_thread_ptr) return;
    egl_release_thread_ptr();

    dlclose(lib);
}
