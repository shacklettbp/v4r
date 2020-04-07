#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>

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
    // the X server like libGLX_nvidia.so. 
    // The below code nulls out the symbol table entry for the
    // __egl_Main in the mesa libEGL library, which prevents the
    // GLVND implementation from seeing it as a valid vendor
    void *lib = dlopen("libEGL_mesa.so", RTLD_LAZY | RTLD_NODELETE);
    if (!lib) return;

    const uint64_t page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page_mask = ~(page_size - 1);

    void *egl_main = dlsym(lib, "__egl_Main");
    Dl_info dl_info;
    ElfW(Sym) *sym_tab_entry;
    [[maybe_unused]] int res =
        dladdr1(egl_main, &dl_info, (void **)&sym_tab_entry, RTLD_DL_SYMENT);
    assert(res != 0);
    void *page = (void *)((uintptr_t)&sym_tab_entry->st_value & page_mask);
    mprotect(page, sizeof(sym_tab_entry->st_value), PROT_READ | PROT_WRITE);
    sym_tab_entry->st_value = 0;
    mprotect(page, sizeof(sym_tab_entry->st_value), PROT_READ);

    dlclose(lib);
}
