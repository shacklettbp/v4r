#include "utils.hpp"

namespace v4r {

[[noreturn]] void fatalExit() noexcept
{
    exit(EXIT_FAILURE);
}

}
