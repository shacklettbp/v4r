#ifndef PROFILER_INL_INCLUDED
#define PROFILER_INL_INCLUDED

#include "profiler.hpp"

namespace v4r {

ElapsedTime::ElapsedTime(ProfileType type)
    : type_(type),
      start_(std::chrono::steady_clock::now()),
      end_()
{}

ElapsedTime::~ElapsedTime()
{
    if (end_.time_since_epoch().count() == 0) {
        end();
    }

    Profiler::report(*this);
}

void ElapsedTime::end()
{
    end_ = std::chrono::steady_clock::now();
}

double ElapsedTime::elapsed() const
{
    std::chrono::duration<double> diff = end_ - start_;
    return std::chrono::duration_cast<std::chrono::seconds>(diff).count();
}

ElapsedTime Profiler::start(ProfileType type)
{
    return ElapsedTime(type);
}

void Profiler::report(const ElapsedTime &time) 
{
    switch (time.getType()) {
    case ProfileType::RenderSubmit:
        stats_.renderSubmit += time.elapsed();
        break;
    case ProfileType::InputSetup:
        stats_.inputSetup += time.elapsed();
        break;
    case ProfileType::CommandRecord:
        stats_.commandRecord += time.elapsed();
        break;
    }
}

Statistics Profiler::getStatistics()
{
    return stats_;
}

}

#endif
