#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace std;

static void rewrite_symbol(void *dst, void *src, uint64_t page_mask)
{
    Dl_info dl_info;
    ElfW(Sym) *sym_tab_entry;
    [[maybe_unused]] int res =
        dladdr1(dst, &dl_info, (void **)&sym_tab_entry, RTLD_DL_SYMENT);
    assert(res != 0);

    void *page = (void *)((uintptr_t)&sym_tab_entry->st_value & page_mask);
    mprotect(page, sizeof(sym_tab_entry->st_value), PROT_READ | PROT_WRITE);
    if (src == nullptr) {
        sym_tab_entry->st_value = 0;
    } else {
        sym_tab_entry->st_value = (uintptr_t)src - (uintptr_t)dl_info.dli_fbase;
    }
    mprotect(page, sizeof(sym_tab_entry->st_value), PROT_READ);
}

static __attribute__((constructor)) void nvidiaLinuxHeadlessHacksEntry()
{
    // The NVIDIA icd entry point is libGLX_nvidia.so.
    // When X11 forwarding is enabled through ssh and DISPLAY is set,
    // vkEnumeratePhysicalDevices crashes with VK_INITIALIZATION_ERROR
    //unsetenv("DISPLAY");

    // When DISPLAY is unset, libGLX_nvidia.so dispatches all calls
    // to libEGL.so. On systems using libGLVND (arch for example),
    // libEGL.so does not know what vendor is running, because there
    // have been no calls to egl and it can't cheat and look at
    // the X server like libGLX_nvidia.so. 
    // The below code nulls out the symbol table entry for the
    // __egl_Main in the mesa libEGL library, which prevents the
    // GLVND implementation from seeing it as a valid vendor
    const uint64_t page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page_mask = ~(page_size - 1);

    void *mesa_egl = dlopen("libEGL_mesa.so", RTLD_LAZY | RTLD_NODELETE);
    void *nvidia_egl = dlopen("libEGL_nvidia.so", RTLD_LAZY | RTLD_NODELETE);
    void *nvidia_glx = dlopen("libGLX_nvidia.so", RTLD_LAZY | RTLD_NODELETE);
    if (!mesa_egl || !nvidia_egl || !nvidia_glx) return;

    void *egl_main = dlsym(mesa_egl, "__egl_Main");
    rewrite_symbol(egl_main, nullptr, page_mask);

    void *egl_negotiate = dlsym(nvidia_egl,
            "vk_icdNegotiateLoaderICDInterfaceVersion");
    void *egl_inst_addr = dlsym(nvidia_egl,
            "vk_icdGetInstanceProcAddr");
    void *egl_phy_addr = dlsym(nvidia_egl,
            "vk_icdGetPhysicalDeviceProcAddr");
    void *glx_negotiate = dlsym(nvidia_glx,
            "vk_icdNegotiateLoaderICDInterfaceVersion");
    void *glx_inst_addr = dlsym(nvidia_glx,
            "vk_icdGetInstanceProcAddr");
    void *glx_phy_addr = dlsym(nvidia_glx,
            "vk_icdGetPhysicalDeviceProcAddr");

    rewrite_symbol(glx_negotiate, egl_negotiate, page_mask);
    rewrite_symbol(glx_inst_addr, egl_inst_addr, page_mask);
    rewrite_symbol(glx_phy_addr, egl_phy_addr, page_mask);

    dlclose(nvidia_glx);
    dlclose(nvidia_egl);
    dlclose(mesa_egl);
}
