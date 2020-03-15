#include "utils.hpp"

namespace v4r {

[[noreturn]] void fatalExit() noexcept
{
    abort();
}

}
