#ifndef _CSTOPWATCH_H_
#define _CSTOPWATCH_H_

#include <windows.h>

#ifdef __cplusplus
typedef struct 
{
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
} stopWatch;

class CStopWatch 
{
private:
	stopWatch timer;
	LARGE_INTEGER frequency;
	double LIToSecs( LARGE_INTEGER & L) const;
public:
	CStopWatch();
  
  void resetTimer();
	void startTimer( );
	void stopTimer( );
	double getElapsedTime() const;
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

int start_stop_watch();
void stop_stop_watch(int id);
double get_duration(int id);

#ifdef __cplusplus
};
#endif
#endif