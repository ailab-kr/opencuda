// helper_timer.h -- the wall-clock stopwatch that programs written for CUDA
// use to time sections (create, start, stop, read milliseconds, delete).
// Written for this project from how the calls are used; times are in
// milliseconds, and a running stopwatch includes the time since its last start.
//
// Used by cu2msl.py only to let the front end parse host code in a .cu file
// whose kernels are being translated.
#ifndef OCU_COMPAT_HELPER_TIMER_H
#define OCU_COMPAT_HELPER_TIMER_H

#include <chrono>

class StopWatchInterface {
public:
    StopWatchInterface() : t0_(0), sum_(0), laps_(0), running_(false) {}
    void start() { t0_ = now(); running_ = true; }
    void stop() { if (running_) { sum_ += now() - t0_; running_ = false; ++laps_; } }
    void reset() { sum_ = 0; laps_ = 0; running_ = false; }
    float getTime() const { return (float)(sum_ + (running_ ? now() - t0_ : 0.0)); }
    float getAverageTime() const { return laps_ ? (float)(sum_ / laps_) : 0.0f; }
private:
    static double now() {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
    }
    double t0_, sum_;
    int laps_;
    bool running_;
};

inline bool sdkCreateTimer(StopWatchInterface **timer) { *timer = new StopWatchInterface(); return true; }
inline bool sdkDeleteTimer(StopWatchInterface **timer) { delete *timer; *timer = 0; return true; }
inline bool sdkStartTimer(StopWatchInterface **timer) { (*timer)->start(); return true; }
inline bool sdkStopTimer(StopWatchInterface **timer) { (*timer)->stop(); return true; }
inline bool sdkResetTimer(StopWatchInterface **timer) { (*timer)->reset(); return true; }
inline float sdkGetTimerValue(StopWatchInterface **timer) { return (*timer)->getTime(); }
inline float sdkGetAverageTimerValue(StopWatchInterface **timer) { return (*timer)->getAverageTime(); }

#endif
