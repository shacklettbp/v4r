#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#include <algorithm>
#include <array>
#include <memory>

namespace v4r {

[[noreturn]] void fatalExit() noexcept;

template <typename>
struct ArraySize {};

template <typename T, size_t N>
struct ArraySize<std::array<T, N>> {
    static constexpr size_t value = N;
};

template <typename T>
class ManagedArray {
public:
    using DeleterFunc = void(void *);
    ManagedArray(T *ptr, DeleterFunc *deleter)
        : ptr_(ptr),
          deleter_(deleter)
    {}

    ManagedArray(const ManagedArray &) = delete;
    ManagedArray(ManagedArray &&o) 
        : ptr_(o.ptr_),
          deleter_(o.deleter_)
    {
        o.ptr_ = nullptr;
    }

    ~ManagedArray()
    {
        if (!ptr_) return;
        deleter_(ptr_);
    }

    constexpr const T& operator[](size_t idx) const noexcept
    {
        return ptr_[idx];
    }

    constexpr T& operator[](size_t idx) noexcept
    {
        return ptr_[idx];
    }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }

private:
    T *ptr_;
    DeleterFunc *deleter_;
};

template <typename T>
class DynArray {
public:
    DynArray(size_t num_elems)
        : data_(std::unique_ptr<T>(new T[num_elems])),
          num_elems_(num_elems)
    {}

    DynArray(const DynArray &) = delete;
    DynArray(DynArray &&) = default;

    constexpr const T& operator[](size_t idx) const noexcept
    {
        return data_.get()[idx];
    }

    constexpr T& operator[](size_t idx) noexcept
    {
        return data_.get()[idx];
    }

    T *data() { return data_.get(); }
    const T *data() const { return data_.get(); }

    constexpr size_t size() const noexcept { return num_elems_; }

    template <typename U>
    class IterBase {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;
        
        explicit IterBase(T *ptr) : ptr_(ptr) {}

        IterBase& operator+=(difference_type d)
        {
            ptr_ += d;
            return *this;
        }

        IterBase& operator-=(difference_type d)
        {
            ptr_ -= d;
            return *this;
        }

        friend IterBase operator+(IterBase it, difference_type d)
        {
            it += d;
            return it;
        }

        friend IterBase operator+(difference_type d, IterBase it)
        {
            return it + d;
        }

        friend IterBase operator-(IterBase it, difference_type d)
        {
            it -= d;
            return it;
        }

        friend difference_type operator-(const IterBase &a,
                                         const IterBase &b)
        {
            return a.ptr_ - b.ptr_;
        }

        bool operator==(IterBase o) const { return ptr_ == o.ptr_; }
        bool operator!=(IterBase o) const { return !(*this == o); }

        reference operator[](difference_type d) const { return ptr_[d]; }
        reference operator*() const { return *ptr_; }

        friend bool operator<(const IterBase &a, const IterBase &b)
        {
            return a.ptr_ < b.ptr_;
        }

        friend bool operator>(const IterBase &a, const IterBase &b)
        {
            return a.ptr_ > b.ptr_;
        }

        friend bool operator<=(const IterBase &a, const IterBase &b)
        {
            return !(a > b);
        }

        friend bool operator>=(const IterBase &a, const IterBase &b)
        {
            return !(a < b);
        }

        IterBase &operator++() { *this += 1; return *this; };
        IterBase &operator--() { *this -= 1; return *this; };

        IterBase operator++(int)
        {
            IterBase t = *this;
            operator++();
            return t;
        }
        IterBase operator--(int)
        {
            IterBase t = *this;
            operator--();
            return t;
        }
    private:
        T *ptr_;
    };

    using iterator = IterBase<T>;
    using const_iterator = IterBase<const T>;

    iterator begin() { return iterator(data_.get()); }
    iterator end() { return iterator(data_.get() + num_elems_); }

    const_iterator begin() const { return const_iterator(data_.get()); }
    const_iterator end() const { return const_iterator(data_.get() + num_elems_); }

private:
    std::unique_ptr<T> data_;
    size_t num_elems_;
};

}

#endif
