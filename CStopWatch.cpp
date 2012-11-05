#include <windows.h>
#include <vector>

#ifndef hr_timer
#include "CStopWatch.h"
#define hr_timer
#endif

double CStopWatch::LIToSecs( LARGE_INTEGER & L) const
{
	return ((double)L.QuadPart /(double)frequency.QuadPart);
}

CStopWatch::CStopWatch()
{
  resetTimer();
}

void CStopWatch::resetTimer()
{
	timer.start.QuadPart=0;
	timer.stop.QuadPart=0;	
	QueryPerformanceFrequency( &frequency );
}

void CStopWatch::startTimer( ) 
{
    QueryPerformanceCounter(&timer.start);
}

void CStopWatch::stopTimer( ) 
{
    QueryPerformanceCounter(&timer.stop);
}


double CStopWatch::getElapsedTime() const
{
	LARGE_INTEGER time;
	time.QuadPart = timer.stop.QuadPart - timer.start.QuadPart;
    return LIToSecs( time) ;
}

//CStopWatch w; // for C access.
std::vector<CStopWatch> watches;
int start_stop_watch()
{
  static int watchKey = -1;

  CStopWatch w;
  watchKey++;
  watches.push_back(w);
  watches[watchKey].startTimer();
  return watchKey;
}
void stop_stop_watch(int id)
{
  watches[id].stopTimer();
}
double get_duration(int id)
{
  return watches[id].getElapsedTime();
}

