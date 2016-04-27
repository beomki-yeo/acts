///////////////////////////////////////////////////////////////////
// IPropagationEngine.h, ACTS project
///////////////////////////////////////////////////////////////////

#ifndef ACTS_EXTRAPOLATIONINTERFACES_IPROPAGATIONENGINE_H
#define ACTS_EXTRAPOLATIONINTERFACES_IPROPAGATIONENGINE_H 1

#include "ACTS/Extrapolation/ExtrapolationCell.h"
#include "ACTS/EventData/TrackParameters.h"
#include "ACTS/EventData/NeutralParameters.h"

namespace Acts {
  
  typedef ExtrapolationCell<TrackParameters>   ExCellCharged;
  typedef ExtrapolationCell<NeutralParameters> ExCellNeutral;

  /** @class IPropagationEngine

      A propagation engine wrapping the propagator algtool it respects the path limit
      to stop particles if needed.

      If the propagation is successful to the surface it will return SuccessfulDestination,
      the parameters will be attached to the ExtrapolationCell as leadParameters,
      such that the engine can chose.

      It also wraps the MultiTrackParameters (@TODO do this actually)

      @author Andreas Salzburger -at - cern.ch
  */

  class IPropagationEngine {

    public:

      /** Virtual destructor */
      virtual ~IPropagationEngine(){}

      /** resolve the boundary situation - for charged particles
          Possible return codes :
           - SuccessPathLimit (path limit reached)
           - SucessDestination (surface hit, only when finalPropagation == true)
           - InProgress (surface hit, when finalPropagation == false)
           - Recovered (surface not hit, leadParameters stay untouched)
      */
      virtual ExtrapolationCode propagate(ExCellCharged& ecCell,
                                          const Surface& sf,
                                          PropDirection dir=alongMomentum,
                                          const BoundaryCheck& bcheck = true,
                                          bool returnCurvilinear = true) const = 0;

      /** resolve the boundary situation - for neutral particles
          Possible return codes :
           - SuccessPathLimit (path limit reached)
           - SucessDestination (surface hit, only when finalPropagation == true)
           - InProgress (surface hit, when finalPropagation == false)
           - Recovered (surface not hit, leadParameters stay untouched)
      */
      virtual ExtrapolationCode propagate(ExCellNeutral& enCell,
                                          const Surface& sf,
                                          PropDirection dir=alongMomentum,
                                          const BoundaryCheck& bcheck = true,
                                          bool returnCurvilinear = true) const = 0;

    protected:
      //!< SCREEN output formatting  (SOP) - unify amongst extrapolation engines
      std::string m_sopPrefix;            //!< prefix for screen output
      std::string m_sopPostfix;           //!< prefix for screen output

  };

} // end of namespace

#endif // ACTS_EXTRAPOLATIONINTERFACES_IPROPAGATIONENGINE_H