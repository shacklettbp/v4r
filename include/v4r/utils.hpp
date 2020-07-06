#ifndef V4R_UTILS_HPP_INCLUDED
#define V4R_UTILS_HPP_INCLUDED

#include <memory>

namespace v4r {

template <typename T>
struct HandleDeleter {
    constexpr HandleDeleter() noexcept = default;
    void operator()(std::remove_extent_t<T> *ptr) const;
};

template <typename T>
using Handle = std::unique_ptr<T, HandleDeleter<T>>;

}

#endif
