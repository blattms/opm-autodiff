// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \copydoc Opm::EclNewtonMethod
 */
#ifndef EWOMS_ECL_NEWTON_METHOD_HH
#define EWOMS_ECL_NEWTON_METHOD_HH

#include <opm/models/blackoil/blackoilnewtonmethod.hh>
#include <opm/models/utils/signum.hh>
#include <opm/common/OpmLog/OpmLog.hpp>


#include <opm/material/common/Unused.hpp>

BEGIN_PROPERTIES

NEW_PROP_TAG(EclNewtonSumTolerance);
NEW_PROP_TAG(EclNewtonStrictIterations);
NEW_PROP_TAG(EclNewtonRelaxedVolumeFraction);
NEW_PROP_TAG(EclNewtonSumToleranceExponent);
NEW_PROP_TAG(EclNewtonRelaxedTolerance);

END_PROPERTIES

namespace Opm {

/*!
 * \brief A newton solver which is ebos specific.
 */
template <class TypeTag>
class EclNewtonMethod : public BlackOilNewtonMethod<TypeTag>
{
    typedef BlackOilNewtonMethod<TypeTag> ParentType;
    typedef typename GET_PROP_TYPE(TypeTag, DiscNewtonMethod) DiscNewtonMethod;

    typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;
    typedef typename GET_PROP_TYPE(TypeTag, SolutionVector) SolutionVector;
    typedef typename GET_PROP_TYPE(TypeTag, GlobalEqVector) GlobalEqVector;
    typedef typename GET_PROP_TYPE(TypeTag, PrimaryVariables) PrimaryVariables;
    typedef typename GET_PROP_TYPE(TypeTag, EqVector) EqVector;
    typedef typename GET_PROP_TYPE(TypeTag, Indices) Indices;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, Linearizer) Linearizer;
    typedef typename GET_PROP_TYPE(TypeTag, ElementContext) ElementContext;

    static const unsigned numEq = GET_PROP_VALUE(TypeTag, NumEq);

    static constexpr int contiSolventEqIdx = Indices::contiSolventEqIdx;
    static constexpr int contiPolymerEqIdx = Indices::contiPolymerEqIdx;
    static constexpr int contiEnergyEqIdx = Indices::contiEnergyEqIdx;

    enum { enableSolvent = GET_PROP_VALUE(TypeTag, EnableSolvent) };
    enum { enablePolymer = GET_PROP_VALUE(TypeTag, EnablePolymer) };
    enum { enablePolymerMolarWeight = GET_PROP_VALUE(TypeTag, EnablePolymerMW) };
    enum { enableFoam = GET_PROP_VALUE(TypeTag, EnableFoam) };
    enum { gasPhaseIdx = FluidSystem::gasPhaseIdx };
    enum { oilPhaseIdx = FluidSystem::oilPhaseIdx };
    enum { waterPhaseIdx = FluidSystem::waterPhaseIdx };
    enum { gasCompIdx = FluidSystem::gasCompIdx };
    enum { oilCompIdx = FluidSystem::oilCompIdx };
    enum { waterCompIdx = FluidSystem::waterCompIdx };

    friend NewtonMethod<TypeTag>;
    friend DiscNewtonMethod;
    friend ParentType;

public:
    EclNewtonMethod(Simulator& simulator) : ParentType(simulator)
    {
        errorPvFraction_ = 1.0;
        relaxedMaxPvFraction_ = EWOMS_GET_PARAM(TypeTag, Scalar, EclNewtonRelaxedVolumeFraction);

        sumTolerance_ = 0.0; // this gets determined in the error calculation proceedure
        relaxedTolerance_ = EWOMS_GET_PARAM(TypeTag, Scalar, EclNewtonRelaxedTolerance);

        numStrictIterations_ = EWOMS_GET_PARAM(TypeTag, int, EclNewtonStrictIterations);

        avgBFactors_.resize(numEq, 0.0);
    }

    /*!
     * \brief Register all run-time parameters for the Newton method.
     */
    static void registerParameters()
    {
        ParentType::registerParameters();

        EWOMS_REGISTER_PARAM(TypeTag, Scalar, EclNewtonSumTolerance,
                             "The maximum error tolerated by the Newton"
                             "method for considering a solution to be "
                             "converged");
        EWOMS_REGISTER_PARAM(TypeTag, int, EclNewtonStrictIterations,
                             "The number of Newton iterations where the"
                             " volumetric error is considered.");
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, EclNewtonRelaxedVolumeFraction,
                             "The fraction of the pore volume of the reservoir "
                             "where the volumetric error may be voilated during "
                             "strict Newton iterations.");
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, EclNewtonSumToleranceExponent,
                             "The the exponent used to scale the sum tolerance by "
                             "the total pore volume of the reservoir.");
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, EclNewtonRelaxedTolerance,
                             "The maximum error which the volumetric residual "
                             "may exhibit if it is in a 'relaxed' "
                             "region during a strict iteration.");
    }

    /*!
     * \brief Returns true if the error of the solution is below the
     *        tolerance.
     */
    bool converged() const
    {
        bool converged = this->simulator_.problem().wellModel().hasWellConverged(avgBFactors_);

        if (errorPvFraction_ < relaxedMaxPvFraction_)
            converged = converged &&  (this->error_ < relaxedTolerance_ && errorSum_ < sumTolerance_);
        // else if (this->numIterations() > numStrictIterations_)
        //     return (this->error_ < relaxedTolerance_ && errorSum_ < sumTolerance_) ;
        else
            converged = converged && (this->error_ <= this->tolerance() && errorSum_ <= sumTolerance_);
        return converged;
    }

    void preSolve_(const SolutionVector& currentSolution  OPM_UNUSED,
                   const GlobalEqVector& currentResidual)
    {
        updateBavg_();

        const auto& constraintsMap = this->model().linearizer().constraintsMap();
        this->lastError_ = this->error_;
        Scalar newtonMaxError = EWOMS_GET_PARAM(TypeTag, Scalar, NewtonMaxError);

        // calculate the error as the maximum weighted tolerance of
        // the solution's residual
        this->error_ = 0.0;
        Dune::FieldVector<Scalar, numEq> componentSumError;
        std::fill(componentSumError.begin(), componentSumError.end(), 0.0);
        Scalar sumPv = 0.0;
        errorPvFraction_ = 0.0;
        const Scalar dt = this->simulator_.timeStepSize();
        std::vector<double> maxCoeff(numEq, std::numeric_limits< Scalar >::lowest() );

        for (unsigned dofIdx = 0; dofIdx < currentResidual.size(); ++dofIdx) {
            // do not consider auxiliary DOFs for the error
            if (dofIdx >= this->model().numGridDof()
                || this->model().dofTotalVolume(dofIdx) <= 0.0)
                continue;

            if (!this->model().isLocalDof(dofIdx))
                continue;

            // also do not consider DOFs which are constraint
            if (this->enableConstraints_()) {
                if (constraintsMap.count(dofIdx) > 0)
                    continue;
            }

            const auto& r = currentResidual[dofIdx];
            Scalar pvValue =
                this->simulator_.problem().referencePorosity(dofIdx, /*timeIdx=*/0)
                * this->model().dofTotalVolume(dofIdx);
            sumPv += pvValue;
            bool cnvViolated = false;

            Scalar dofVolume = this->model().dofTotalVolume(dofIdx);

            for (unsigned eqIdx = 0; eqIdx < r.size(); ++eqIdx) {
                double factor = 1.0;
                Scalar CNV = r[eqIdx] * dt * avgBFactors_[eqIdx] / pvValue;
                Scalar MB = r[eqIdx] * avgBFactors_[eqIdx];

                // in the case of a volumetric formulation, the residual in the above is
                // per cubic meter
                if (GET_PROP_VALUE(TypeTag, UseVolumetricResidual)) {
                    CNV *= dofVolume;
                    MB *= dofVolume;
                    factor = dofVolume;
                }

                maxCoeff[eqIdx] = std::max(maxCoeff[eqIdx], std::abs(r[eqIdx])*factor/pvValue);
                this->error_ = Opm::max(std::abs(CNV), this->error_);

                if (std::abs(CNV) > this->tolerance_)
                    cnvViolated = true;

                componentSumError[eqIdx] += std::abs(MB);
            }
            if (cnvViolated)
                errorPvFraction_ += pvValue;
        }

        this->comm_.max(maxCoeff.data(), maxCoeff.size());
        // take the other processes into account
        this->error_ = this->comm_.max(this->error_);
        std::vector<double> CNVV(numEq);
        this->error_ = std::numeric_limits< Scalar >::lowest();

        for (unsigned eqIdx = 0; eqIdx < numEq; ++eqIdx)
        {
            CNVV[eqIdx] = avgBFactors_[eqIdx] * dt * maxCoeff[eqIdx];
            this->error_ = std::max(this->error_ , CNVV[eqIdx]);
        }

        componentSumError = this->comm_.sum(componentSumError);
        sumPv = this->comm_.sum(sumPv);
        errorPvFraction_ = this->comm_.sum(errorPvFraction_);

        componentSumError /= sumPv;
        componentSumError *= dt;

        errorPvFraction_ /= sumPv;

        errorSum_ = 0;
        for (unsigned eqIdx = 0; eqIdx < numEq; ++eqIdx)
            errorSum_ = std::max(std::abs(componentSumError[eqIdx]), errorSum_);

        // scale the tolerance for the total error with the pore volume. by default, the
        // exponent is 1/3, i.e., cubic root.
        Scalar x = EWOMS_GET_PARAM(TypeTag, Scalar, EclNewtonSumTolerance);
        Scalar y = EWOMS_GET_PARAM(TypeTag, Scalar, EclNewtonSumToleranceExponent);
        sumTolerance_ = x;//*std::pow(sumPv, y);

        this->endIterMsg() << " (max: " << this->tolerance_ << ", violated for " << errorPvFraction_*100 << "% of the pore volume), aggegate error: " << errorSum_ << " (max: " << sumTolerance_ << ")";

        // make sure that the error never grows beyond the maximum
        // allowed one
        if (this->error_ > newtonMaxError)
            throw Opm::NumericalIssue("Newton: Error "+std::to_string(double(this->error_))
                                        +" is larger than maximum allowed error of "
                                        +std::to_string(double(newtonMaxError)));

        // make sure that the error never grows beyond the maximum
        // allowed one
        if (errorSum_ > newtonMaxError)
            throw Opm::NumericalIssue("Newton: Sum of the error "+std::to_string(double(errorSum_))
                                        +" is larger than maximum allowed error of "
                                        +std::to_string(double(newtonMaxError)));
    }

    void endIteration_(SolutionVector& nextSolution,
                       const SolutionVector& currentSolution)
    {
        ParentType::endIteration_(nextSolution, currentSolution);
        OpmLog::debug( "Newton iteration " + std::to_string(this->numIterations_) + ""
                  + " error: " + std::to_string(double(this->error_))
                  + this->endIterMsg().str());
        this->endIterMsg().str("");
    }

    Scalar getavgBFactors(size_t eqIdx) const {
        return avgBFactors_[eqIdx];
    }

private:

    bool updateBavg_()
    {
        ElementContext elemCtx(this->simulator_);
        const auto& vanguard = this->simulator_.vanguard();
        auto elemIt = vanguard.gridView().template begin</*codim=*/0, Dune::Interior_Partition>();
        const auto& elemEndIt = vanguard.gridView().template end</*codim=*/0, Dune::Interior_Partition>();
        std::vector<Scalar> B_avg(numEq,1.0);
        for (; elemIt != elemEndIt; ++elemIt) {
            const auto& elem = *elemIt;

            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
            const auto& iq = elemCtx.intensiveQuantities(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& fs = iq.fluidState();
            B_avg[Indices::conti0EqIdx + FluidSystem::waterCompIdx] += 1.0 / fs.invB(waterPhaseIdx).value();
            B_avg[Indices::conti0EqIdx + FluidSystem::gasCompIdx] += 1.0 / fs.invB(gasPhaseIdx).value();
            B_avg[Indices::conti0EqIdx + FluidSystem::oilCompIdx] += 1.0 / fs.invB(oilPhaseIdx).value();
            if (enableSolvent)
                B_avg[ Indices::contiSolventEqIdx ] += 1.0 / iq.solventInverseFormationVolumeFactor().value();
            if (enablePolymer)
                B_avg[ Indices::contiPolymerEqIdx ] += 1.0 / fs.invB(waterPhaseIdx).value();
            if (enableFoam)
                B_avg[ Indices::contiFoamEqIdx ] += 1.0 / fs.invB(gasPhaseIdx).value();
            if (enablePolymerMolarWeight)
                B_avg[ Indices::contiPolymerMWEqIdx ] += 1.0 / fs.invB(waterPhaseIdx).value();
        }
        for (size_t i = 0; i < numEq; ++i) {
            B_avg[i] = this->comm_.sum(B_avg[i]);
        }
        int totalNumGridDof = this->model().numGridDof();
        totalNumGridDof = this->comm_.sum(totalNumGridDof);

        for (size_t i = 0; i < numEq; ++i)
            avgBFactors_[i] = B_avg[i] / totalNumGridDof ;

        return true;
    }

    Scalar errorPvFraction_;
    Scalar errorSum_;

    Scalar relaxedTolerance_;
    Scalar relaxedMaxPvFraction_;

    Scalar sumTolerance_;

    std::vector<Scalar> avgBFactors_;
    int numStrictIterations_;
};
} // namespace Opm

#endif
