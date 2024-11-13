
#include "SimStreamer.h"
#include "JPEGSamples.h"


#ifdef INCLUDE_SIMDATA
SimStreamer::SimStreamer(SOCKET aClient, bool showBig) : CStreamer(aClient, showBig ? 400 : 320, showBig ? 300 : 240)
{
    m_showBig = showBig;
}

void SimStreamer::streamImage(uint32_t curMsec)
{
    if(m_showBig) {
        BufPtr bytes = capture_jpg;
        uint32_t len = capture_jpg_len;

        streamFrame(bytes, len, curMsec);
    }
    else {
        BufPtr bytes = octo_jpg;
        uint32_t len = octo_jpg_len;

        streamFrame(bytes, len, curMsec);
    }
}

#endif
