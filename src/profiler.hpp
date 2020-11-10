#ifndef V4R_PROFILER_HPP_INCLUDED
#define V4R_PROFILER_HPP_INCLUDED

#include <v4r/stats.hpp>

#include <chrono>

namespace v4r {

enum class ProfileType {
    RenderSubmit,
    InputSetup,
    CommandRecord,
};

class ElapsedTime {
public:
    inline ElapsedTime(ProfileType type);
    inline ~ElapsedTime();

    inline void end();

    inline double elapsed() const;

    inline ProfileType getType() const { return type_; }

private:
    ProfileType type_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
    std::chrono::time_point<std::chrono::steady_clock> end_;
};

class Profiler {
public:
    static inline ElapsedTime start(ProfileType type);
    static inline void report(const ElapsedTime &time);

    static inline Statistics getStatistics();

private:
    static Statistics stats_;
};

}

#ifndef PROFILER_INL_INCLUDED
#include "profiler.inl"
#endif

#endif
