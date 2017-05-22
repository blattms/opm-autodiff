/*
  Copyright 2013, 2015 SINTEF ICT, Applied Mathematics.
  Copyright 2015 Andreas Lauser

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_SIMULATORFULLYIMPLICITBLACKOILEBOS_HEADER_INCLUDED
#define OPM_SIMULATORFULLYIMPLICITBLACKOILEBOS_HEADER_INCLUDED

//#include <opm/autodiff/SimulatorBase.hpp>
//#include <opm/autodiff/SimulatorFullyImplicitBlackoilOutputEbos.hpp>
#include <opm/autodiff/SimulatorFullyImplicitBlackoilOutput.hpp>
#include <opm/autodiff/IterationReport.hpp>
#include <opm/autodiff/NonlinearSolver.hpp>
#include <opm/autodiff/BlackoilModelEbos.hpp>
#include <opm/autodiff/BlackoilModelParameters.hpp>
#include <opm/autodiff/WellStateFullyImplicitBlackoilDense.hpp>
#include <opm/autodiff/StandardWellsDense.hpp>
#include <opm/autodiff/RateConverter.hpp>
#include <opm/autodiff/SimFIBODetails.hpp>
#include <opm/autodiff/moduleVersion.hpp>
#include <opm/simulators/timestepping/AdaptiveTimeStepping.hpp>
#include <opm/core/utility/initHydroCarbonState.hpp>
#include <opm/core/utility/StopWatch.hpp>

#include <opm/common/Exceptions.hpp>
#include <opm/common/ErrorMacros.hpp>

#include <dune/common/unused.hh>

namespace Opm {

class SimulatorFullyImplicitBlackoilEbos;
//class StandardWellsDense<FluidSystem>;

/// a simulator for the blackoil model
class SimulatorFullyImplicitBlackoilEbos
{
public:
    typedef typename TTAG(EclFlowProblem) TypeTag;
    typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;
    typedef typename GET_PROP_TYPE(TypeTag, Grid) Grid;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;
    typedef typename GET_PROP_TYPE(TypeTag, ElementContext) ElementContext;
    typedef typename GET_PROP_TYPE(TypeTag, Indices) BlackoilIndices;
    typedef typename GET_PROP_TYPE(TypeTag, PrimaryVariables)  PrimaryVariables;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLaw) MaterialLaw;


    typedef WellStateFullyImplicitBlackoilDense WellState;
    typedef BlackoilState ReservoirState;
    typedef BlackoilOutputWriter OutputWriter;
    typedef BlackoilModelEbos Model;
    typedef BlackoilModelParameters ModelParameters;
    typedef NonlinearSolver<Model> Solver;
    typedef StandardWellsDense<TypeTag> WellModel;


    /// Initialise from parameters and objects to observe.
    /// \param[in] param       parameters, this class accepts the following:
    ///     parameter (default)            effect
    ///     -----------------------------------------------------------
    ///     output (true)                  write output to files?
    ///     output_dir ("output")          output directoty
    ///     output_interval (1)            output every nth step
    ///     nl_pressure_residual_tolerance (0.0) pressure solver residual tolerance (in Pascal)
    ///     nl_pressure_change_tolerance (1.0)   pressure solver change tolerance (in Pascal)
    ///     nl_pressure_maxiter (10)       max nonlinear iterations in pressure
    ///     nl_maxiter (30)                max nonlinear iterations in transport
    ///     nl_tolerance (1e-9)            transport solver absolute residual tolerance
    ///     num_transport_substeps (1)     number of transport steps per pressure step
    ///     use_segregation_split (false)  solve for gravity segregation (if false,
    ///                                    segregation is ignored).
    ///
    /// \param[in] props         fluid and rock properties
    /// \param[in] linsolver     linear solver
    /// \param[in] has_disgas    true for dissolved gas option
    /// \param[in] has_vapoil    true for vaporized oil option
    /// \param[in] eclipse_state the object which represents an internalized ECL deck
    /// \param[in] output_writer
    /// \param[in] threshold_pressures_by_face   if nonempty, threshold pressures that inhibit flow
    SimulatorFullyImplicitBlackoilEbos(Simulator& ebosSimulator,
                                       const ParameterGroup& param,
                                       BlackoilPropsAdFromDeck& props,
                                       NewtonIterationBlackoilInterface& linsolver,
                                       const bool has_disgas,
                                       const bool has_vapoil,
                                       const EclipseState& /* eclState */,
                                       OutputWriter& output_writer,
                                       const std::unordered_set<std::string>& defunct_well_names)
        : ebosSimulator_(ebosSimulator),
          param_(param),
          model_param_(param),
          solver_param_(param),
          props_(props),
          solver_(linsolver),
          has_disgas_(has_disgas),
          has_vapoil_(has_vapoil),
          terminal_output_(param.getDefault("output_terminal", true)),
          output_writer_(output_writer),
          defunct_well_names_( defunct_well_names ),
          is_parallel_run_( false )
    {
        extractLegacyCellPvtRegionIndex_();
        rateConverter_.reset(new RateConverterType(props.phaseUsage(),
                                                   legacyCellPvtRegionIdx_.data(),
                                                   AutoDiffGrid::numCells(grid()),
                                                   std::vector<int>(AutoDiffGrid::numCells(grid()), 0)));

#if HAVE_MPI
        if ( solver_.parallelInformation().type() == typeid(ParallelISTLInformation) )
        {
            const ParallelISTLInformation& info =
                boost::any_cast<const ParallelISTLInformation&>(solver_.parallelInformation());
            // Only rank 0 does print to std::cout
            terminal_output_ = terminal_output_ && ( info.communicator().rank() == 0 );
            is_parallel_run_ = ( info.communicator().size() > 1 );
        }
#endif
    }

    /// Run the simulation.
    /// This will run succesive timesteps until timer.done() is true. It will
    /// modify the reservoir and well states.
    /// \param[in,out] timer       governs the requested reporting timesteps
    /// \param[in,out] state       state of reservoir: pressure, fluxes
    /// \return                    simulation report, with timing data
    SimulatorReport run(SimulatorTimer& timer,
                        ReservoirState& state)
    {
        WellState prev_well_state;

        ExtraData extra;

        failureReport_ = SimulatorReport();
        extractLegacyPoreVolume_();
        extractLegacyDepth_();

        if (output_writer_.isRestart()) {
            // This is a restart, populate WellState and ReservoirState state objects from restart file
            output_writer_.initFromRestartFile(props_.phaseUsage(), grid(), state, prev_well_state, extra);
            initHydroCarbonState(state, props_.phaseUsage(), Opm::UgGridHelpers::numCells(grid()), has_disgas_, has_vapoil_);
            initHysteresisParams(state);
        }

        // Create timers and file for writing timing info.
        Opm::time::StopWatch solver_timer;
        Opm::time::StopWatch step_timer;
        Opm::time::StopWatch total_timer;
        total_timer.start();
        std::string tstep_filename = output_writer_.outputDirectory() + "/step_timing.txt";
        std::ofstream tstep_os(tstep_filename.c_str());

        const auto& schedule = eclState().getSchedule();

        // adaptive time stepping
        const auto& events = schedule.getEvents();
        std::unique_ptr< AdaptiveTimeStepping > adaptiveTimeStepping;
        if( param_.getDefault("timestep.adaptive", true ) )
        {

            if (param_.getDefault("use_TUNING", false)) {
                adaptiveTimeStepping.reset( new AdaptiveTimeStepping( schedule.getTuning(), timer.currentStepNum(), param_, terminal_output_ ) );
            } else {
                adaptiveTimeStepping.reset( new AdaptiveTimeStepping( param_, terminal_output_ ) );
            }

            if (output_writer_.isRestart()) {
                if (extra.suggested_step > 0.0) {
                    adaptiveTimeStepping->setSuggestedNextStep(extra.suggested_step);
                }
            }
        }

        std::string restorefilename = param_.getDefault("restorefile", std::string("") );
        if( ! restorefilename.empty() )
        {
            // -1 means that we'll take the last report step that was written
            const int desiredRestoreStep = param_.getDefault("restorestep", int(-1) );

            output_writer_.restore( timer,
                                    state,
                                    prev_well_state,
                                    restorefilename,
                                    desiredRestoreStep );
        }

        DynamicListEconLimited dynamic_list_econ_limited;
        SimulatorReport report;
        SimulatorReport stepReport;

        std::vector<int> fipnum_global = eclState().get3DProperties().getIntGridProperty("FIPNUM").getData();
        //Get compressed cell fipnum.
        std::vector<int> fipnum(Opm::UgGridHelpers::numCells(grid()));
        if (fipnum_global.empty()) {
            std::fill(fipnum.begin(), fipnum.end(), 0);
        } else {
            for (size_t c = 0; c < fipnum.size(); ++c) {
                fipnum[c] = fipnum_global[Opm::UgGridHelpers::globalCell(grid())[c]];
            }
        }
        std::vector<std::vector<double>> originalFluidInPlace;
        std::vector<double> originalFluidInPlaceTotals;

        if ( model_param_.matrix_add_well_contributions_ )
        {
            ebosSimulator_.model().clearAuxiliaryModules();
            auto auxMod = std::make_shared<WellConnectionAuxiliaryModule<TypeTag> >(eclState(), grid());
            ebosSimulator_.model().addAuxiliaryModule(auxMod);
        }

        // Main simulation loop.
        while (!timer.done()) {
            // Report timestep.
            step_timer.start();
            if ( terminal_output_ )
            {
                std::ostringstream ss;
                timer.report(ss);
                OpmLog::note(ss.str());
            }

            // Create wells and well state.
            WellsManager wells_manager(eclState(),
                                       timer.currentStepNum(),
                                       Opm::UgGridHelpers::numCells(grid()),
                                       Opm::UgGridHelpers::globalCell(grid()),
                                       Opm::UgGridHelpers::cartDims(grid()),
                                       Opm::UgGridHelpers::dimensions(grid()),
                                       Opm::UgGridHelpers::cell2Faces(grid()),
                                       Opm::UgGridHelpers::beginFaceCentroids(grid()),
                                       dynamic_list_econ_limited,
                                       is_parallel_run_,
                                       defunct_well_names_ );
            const Wells* wells = wells_manager.c_wells();
            WellState well_state;
            well_state.init(wells, state, prev_well_state, props_.phaseUsage());

            // give the polymer and surfactant simulators the chance to do their stuff
            handleAdditionalWellInflow(timer, wells_manager, well_state, wells);

            // Compute reservoir volumes for RESV controls.
            computeRESV(timer.currentStepNum(), wells, state, well_state);

            // Run a multiple steps of the solver depending on the time step control.
            solver_timer.start();

            WellModel well_model(wells, &(wells_manager.wellCollection()), model_param_, terminal_output_);
            auto solver = createSolver(well_model);

            std::vector<std::vector<double>> currentFluidInPlace;
            std::vector<double> currentFluidInPlaceTotals;

            // Compute orignal fluid in place if this has not been done yet
            if (originalFluidInPlace.empty()) {
                solver->model().convertInput(/*iterationIdx=*/0, state, ebosSimulator_ );
                ebosSimulator_.model().invalidateIntensiveQuantitiesCache(/*timeIdx=*/0);

                originalFluidInPlace = solver->computeFluidInPlace(fipnum);
                originalFluidInPlaceTotals = FIPTotals(originalFluidInPlace, state);
                FIPUnitConvert(eclState().getUnits(), originalFluidInPlace);
                FIPUnitConvert(eclState().getUnits(), originalFluidInPlaceTotals);

                currentFluidInPlace = originalFluidInPlace;
                currentFluidInPlaceTotals = originalFluidInPlaceTotals;
            }

            // write the inital state at the report stage
            if (timer.initialStep()) {
                Dune::Timer perfTimer;
                perfTimer.start();

                if (terminal_output_) {
                    outputFluidInPlace(originalFluidInPlaceTotals, currentFluidInPlaceTotals,eclState().getUnits(), 0);
                    for (size_t reg = 0; reg < originalFluidInPlace.size(); ++reg) {
                        outputFluidInPlace(originalFluidInPlace[reg], currentFluidInPlace[reg], eclState().getUnits(), reg+1);
                    }
                }

                // No per cell data is written for initial step, but will be
                // for subsequent steps, when we have started simulating
                output_writer_.writeTimeStep( timer, state, well_state, solver->model() );

                report.output_write_time += perfTimer.stop();
            }

            if( terminal_output_ )
            {
                std::ostringstream step_msg;
                boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%d-%b-%Y");
                step_msg.imbue(std::locale(std::locale::classic(), facet));
                step_msg << "\nTime step " << std::setw(4) <<timer.currentStepNum()
                         << " at day " << (double)unit::convert::to(timer.simulationTimeElapsed(), unit::day)
                         << "/" << (double)unit::convert::to(timer.totalTime(), unit::day)
                         << ", date = " << timer.currentDateTime();
                OpmLog::info(step_msg.str());
            }

            solver->model().beginReportStep();

            // If sub stepping is enabled allow the solver to sub cycle
            // in case the report steps are too large for the solver to converge
            //
            // \Note: The report steps are met in any case
            // \Note: The sub stepping will require a copy of the state variables
            if( adaptiveTimeStepping ) {
                bool event = events.hasEvent(ScheduleEvents::NEW_WELL, timer.currentStepNum()) ||
                        events.hasEvent(ScheduleEvents::PRODUCTION_UPDATE, timer.currentStepNum()) ||
                        events.hasEvent(ScheduleEvents::INJECTION_UPDATE, timer.currentStepNum()) ||
                        events.hasEvent(ScheduleEvents::WELL_STATUS_CHANGE, timer.currentStepNum());
                stepReport = adaptiveTimeStepping->step( timer, *solver, state, well_state, event, output_writer_,
                                                         output_writer_.requireFIPNUM() ? &fipnum : nullptr );
                report += stepReport;
                failureReport_ += adaptiveTimeStepping->failureReport();
            }
            else {
                // solve for complete report step
                stepReport = solver->step(timer, state, well_state);
                report += stepReport;
                failureReport_ += solver->failureReport();

                if( terminal_output_ )
                {
                    //stepReport.briefReport();
                    std::ostringstream iter_msg;
                    iter_msg << "Stepsize " << (double)unit::convert::to(timer.currentStepLength(), unit::day);
                    if (solver->wellIterations() != 0) {
                        iter_msg << " days well iterations = " << solver->wellIterations() << ", ";
                    }
                    iter_msg << "non-linear iterations = " << solver->nonlinearIterations()
                             << ", total linear iterations = " << solver->linearIterations()
                             << "\n";
                    OpmLog::info(iter_msg.str());
                }
            }

            solver->model().endReportStep();

            // take time that was used to solve system for this reportStep
            solver_timer.stop();

            // update timing.
            report.solver_time += solver_timer.secsSinceStart();

            if ( output_writer_.output() ) {
                if ( output_writer_.isIORank() )
                {
                    stepReport.reportParam(tstep_os);
                }
            }

            // Increment timer, remember well state.
            ++timer;

            // Compute current fluid in place.
            currentFluidInPlace = solver->computeFluidInPlace(fipnum);
            currentFluidInPlaceTotals = FIPTotals(currentFluidInPlace, state);

            const std::string version = moduleVersionName();

            FIPUnitConvert(eclState().getUnits(), currentFluidInPlace);
            FIPUnitConvert(eclState().getUnits(), currentFluidInPlaceTotals);

            if (terminal_output_ )
            {
                outputTimestampFIP(timer, version);
                outputFluidInPlace(originalFluidInPlaceTotals, currentFluidInPlaceTotals,eclState().getUnits(), 0);
                for (size_t reg = 0; reg < originalFluidInPlace.size(); ++reg) {
                    outputFluidInPlace(originalFluidInPlace[reg], currentFluidInPlace[reg], eclState().getUnits(), reg+1);
                }

                std::string msg;
                msg =
                    "Time step took " + std::to_string(solver_timer.secsSinceStart()) + " seconds; "
                    "total solver time " + std::to_string(report.solver_time) + " seconds.";
                OpmLog::note(msg);
            }

            // write simulation state at the report stage
            Dune::Timer perfTimer;
            perfTimer.start();
            const double nextstep = adaptiveTimeStepping ? adaptiveTimeStepping->suggestedNextStep() : -1.0;
            output_writer_.writeTimeStep( timer, state, well_state, solver->model(), false, nextstep );
            report.output_write_time += perfTimer.stop();

            prev_well_state = well_state;

            updateListEconLimited(solver, eclState().getSchedule(), timer.currentStepNum(), wells,
                                  well_state, dynamic_list_econ_limited);
        }

        // Stop timer and create timing report
        total_timer.stop();
        report.total_time = total_timer.secsSinceStart();
        report.converged = true;
        return report;
    }

    /** \brief Returns the simulator report for the failed substeps of the simulation.
     */
    const SimulatorReport& failureReport() const { return failureReport_; };

    const Grid& grid() const
    { return ebosSimulator_.gridManager().grid(); }

protected:
    void handleAdditionalWellInflow(SimulatorTimer& /* timer */,
                                    WellsManager& /* wells_manager */,
                                    WellState& /* well_state */,
                                    const Wells* /* wells */)
    { }

    std::unique_ptr<Solver> createSolver(WellModel& well_model)
    {
        const auto& gridView = ebosSimulator_.gridView();
        const PhaseUsage& phaseUsage = props_.phaseUsage();
        const std::vector<bool> activePhases = detail::activePhases(phaseUsage);
        const double gravity = ebosSimulator_.problem().gravity()[2];

        // calculate the number of elements of the compressed sequential grid. this needs
        // to be done in two steps because the dune communicator expects a reference as
        // argument for sum()
        int globalNumCells = gridView.size(/*codim=*/0);
        globalNumCells = gridView.comm().sum(globalNumCells);

        well_model.init(phaseUsage,
                        activePhases,
                        /*vfpProperties=*/nullptr,
                        gravity,
                        legacyDepth_,
                        legacyPoreVolume_,
                        rateConverter_.get(),
                        globalNumCells);
        auto model = std::unique_ptr<Model>(new Model(ebosSimulator_,
                                                      model_param_,
                                                      props_,
                                                      well_model,
                                                      solver_,
                                                      terminal_output_));

        return std::unique_ptr<Solver>(new Solver(solver_param_, std::move(model)));
    }

    void computeRESV(const std::size_t step,
                     const Wells* wells,
                     const BlackoilState& x,
                     WellState& xw)
    {
        typedef SimFIBODetails::WellMap WellMap;

        const auto w_ecl = eclState().getSchedule().getWells(step);
        const WellMap& wmap = SimFIBODetails::mapWells(w_ecl);

        const std::vector<int>& resv_wells = SimFIBODetails::resvWells(wells, step, wmap);

        const std::size_t number_resv_wells        = resv_wells.size();
        std::size_t       global_number_resv_wells = number_resv_wells;
#if HAVE_MPI
        if ( solver_.parallelInformation().type() == typeid(ParallelISTLInformation) )
        {
            const auto& info =
                boost::any_cast<const ParallelISTLInformation&>(solver_.parallelInformation());
            global_number_resv_wells = info.communicator().sum(global_number_resv_wells);
            if ( global_number_resv_wells )
            {
                // At least one process has resv wells. Therefore rate converter needs
                // to calculate averages over regions that might cross process
                // borders. This needs to be done by all processes and therefore
                // outside of the next if statement.
                rateConverter_->defineState(x, boost::any_cast<const ParallelISTLInformation&>(solver_.parallelInformation()));
            }
        }
        else
#endif
        {
            if ( global_number_resv_wells )
            {
                rateConverter_->defineState(x);
            }
        }

        if (! resv_wells.empty()) {
            const PhaseUsage&                    pu = props_.phaseUsage();
            const std::vector<double>::size_type np = props_.numPhases();

            std::vector<double> distr (np);
            std::vector<double> hrates(np);
            std::vector<double> prates(np);

            for (std::vector<int>::const_iterator
                     rp = resv_wells.begin(), e = resv_wells.end();
                 rp != e; ++rp)
            {
                WellControls* ctrl = wells->ctrls[*rp];
                const bool is_producer = wells->type[*rp] == PRODUCER;

                // RESV control mode, all wells
                {
                    const int rctrl = SimFIBODetails::resv_control(ctrl);

                    if (0 <= rctrl) {
                        const std::vector<double>::size_type off = (*rp) * np;

                        if (is_producer) {
                            // Convert to positive rates to avoid issues
                            // in coefficient calculations.
                            std::transform(xw.wellRates().begin() + (off + 0*np),
                                           xw.wellRates().begin() + (off + 1*np),
                                           prates.begin(), std::negate<double>());
                        } else {
                            std::copy(xw.wellRates().begin() + (off + 0*np),
                                      xw.wellRates().begin() + (off + 1*np),
                                      prates.begin());
                        }

                        const int fipreg = 0; // Hack.  Ignore FIP regions.
                        rateConverter_->calcCoeff(prates, fipreg, distr);

                        well_controls_iset_distr(ctrl, rctrl, & distr[0]);
                    }
                }

                // RESV control, WCONHIST wells.  A bit of duplicate
                // work, regrettably.
                if (is_producer && wells->name[*rp] != 0) {
                    WellMap::const_iterator i = wmap.find(wells->name[*rp]);

                    if (i != wmap.end()) {
                        const auto* wp = i->second;

                        const WellProductionProperties& p =
                            wp->getProductionProperties(step);

                        if (! p.predictionMode) {
                            // History matching (WCONHIST/RESV)
                            SimFIBODetails::historyRates(pu, p, hrates);

                            const int fipreg = 0; // Hack.  Ignore FIP regions.
                            rateConverter_->calcCoeff(hrates, fipreg, distr);

                            // WCONHIST/RESV target is sum of all
                            // observed phase rates translated to
                            // reservoir conditions.  Recall sign
                            // convention: Negative for producers.
                            const double target =
                                - std::inner_product(distr.begin(), distr.end(),
                                                     hrates.begin(), 0.0);

                            well_controls_clear(ctrl);
                            well_controls_assert_number_of_phases(ctrl, int(np));

                            static const double invalid_alq = -std::numeric_limits<double>::max();
                            static const int invalid_vfp = -std::numeric_limits<int>::max();

                            const int ok_resv =
                                well_controls_add_new(RESERVOIR_RATE, target,
                                                      invalid_alq, invalid_vfp,
                                                      & distr[0], ctrl);

                            // For WCONHIST the BHP limit is set to 1 atm.
                            // or a value specified using WELTARG
                            double bhp_limit = (p.BHPLimit > 0) ? p.BHPLimit : unit::convert::from(1.0, unit::atm);
                            const int ok_bhp =
                                well_controls_add_new(BHP, bhp_limit,
                                                      invalid_alq, invalid_vfp,
                                                      NULL, ctrl);

                            if (ok_resv != 0 && ok_bhp != 0) {
                                xw.currentControls()[*rp] = 0;
                                well_controls_set_current(ctrl, 0);
                            }
                        }
                    }
                }
            }
        }

        if( wells )
        {
            for (int w = 0, nw = wells->number_of_wells; w < nw; ++w) {
                WellControls* ctrl = wells->ctrls[w];
                const bool is_producer = wells->type[w] == PRODUCER;
                if (!is_producer && wells->name[w] != 0) {
                    WellMap::const_iterator i = wmap.find(wells->name[w]);
                    if (i != wmap.end()) {
                        const auto* wp = i->second;
                        const WellInjectionProperties& injector = wp->getInjectionProperties(step);
                        if (!injector.predictionMode) {
                            //History matching WCONINJEH
                            static const double invalid_alq = -std::numeric_limits<double>::max();
                            static const int invalid_vfp = -std::numeric_limits<int>::max();
                            // For WCONINJEH the BHP limit is set to a large number
                            // or a value specified using WELTARG
                            double bhp_limit = (injector.BHPLimit > 0) ? injector.BHPLimit : std::numeric_limits<double>::max();
                            const int ok_bhp =
                                well_controls_add_new(BHP, bhp_limit,
                                                      invalid_alq, invalid_vfp,
                                                      NULL, ctrl);
                            if (!ok_bhp) {
                                OPM_THROW(std::runtime_error, "Failed to add well control.");
                            }
                        }
                    }
                }
            }
        }
    }



    void updateListEconLimited(const std::unique_ptr<Solver>& solver,
                               const Schedule& schedule,
                               const int current_step,
                               const Wells* wells,
                               const WellState& well_state,
                               DynamicListEconLimited& list_econ_limited) const
    {
        solver->model().wellModel().updateListEconLimited(schedule, current_step, wells,
                                                          well_state, list_econ_limited);
    }

    void FIPUnitConvert(const UnitSystem& units,
                        std::vector<std::vector<double>>& fip)
    {
        for (size_t i = 0; i < fip.size(); ++i) {
            FIPUnitConvert(units, fip[i]);
        }
    }


    void FIPUnitConvert(const UnitSystem& units,
                        std::vector<double>& fip)
    {
        if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_FIELD) {
            fip[0] = unit::convert::to(fip[0], unit::stb);
            fip[1] = unit::convert::to(fip[1], unit::stb);
            fip[2] = unit::convert::to(fip[2], 1000*unit::cubic(unit::feet));
            fip[3] = unit::convert::to(fip[3], 1000*unit::cubic(unit::feet));
            fip[4] = unit::convert::to(fip[4], unit::stb);
            fip[5] = unit::convert::to(fip[5], unit::stb);
            fip[6] = unit::convert::to(fip[6], unit::psia);
        }
        else if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_METRIC) {
            fip[6] = unit::convert::to(fip[6], unit::barsa);
        }
        else {
            OPM_THROW(std::runtime_error, "Unsupported unit type for fluid in place output.");
        }
    }


    std::vector<double> FIPTotals(const std::vector<std::vector<double>>& fip, const ReservoirState& /* state */)
    {
        std::vector<double> totals(7,0.0);
        for (int i = 0; i < 5; ++i) {
            for (size_t reg = 0; reg < fip.size(); ++reg) {
                totals[i] += fip[reg][i];
            }
        }

        const auto& gridView = ebosSimulator_.gridManager().gridView();
        const auto& comm = gridView.comm();
        double pv_hydrocarbon_sum = 0.0;
        double p_pv_hydrocarbon_sum = 0.0;

        ElementContext elemCtx(ebosSimulator_);
        const auto& elemEndIt = gridView.template end</*codim=*/0>();
        for (auto elemIt = gridView.template begin</*codim=*/0>();
             elemIt != elemEndIt;
             ++elemIt)
        {
            const auto& elem = *elemIt;
            if (elem.partitionType() != Dune::InteriorEntity) {
                continue;
            }

            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);

            const unsigned cellIdx = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& intQuants = elemCtx.intensiveQuantities(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& fs = intQuants.fluidState();

            const double p = fs.pressure(FluidSystem::oilPhaseIdx).value();
            const double hydrocarbon = fs.saturation(FluidSystem::oilPhaseIdx).value() + fs.saturation(FluidSystem::gasPhaseIdx).value();

            // calculate the pore volume of the current cell. Note that the
            // porosity returned by the intensive quantities is defined as the
            // ratio of pore space to total cell volume and includes all pressure
            // dependent (-> rock compressibility) and static modifiers (MULTPV,
            // MULTREGP, NTG, PORV, MINPV and friends). Also note that because of
            // this, the porosity returned by the intensive quantities can be
            // outside of the physical range [0, 1] in pathetic cases.
            const double pv =
                ebosSimulator_.model().dofTotalVolume(cellIdx)
                * intQuants.porosity().value();

            totals[5] += pv;
            pv_hydrocarbon_sum += pv*hydrocarbon;
            p_pv_hydrocarbon_sum += p*pv*hydrocarbon;
        }

        pv_hydrocarbon_sum = comm.sum(pv_hydrocarbon_sum);
        p_pv_hydrocarbon_sum = comm.sum(p_pv_hydrocarbon_sum);
        totals[5] = comm.sum(totals[5]);
        totals[6] = (p_pv_hydrocarbon_sum / pv_hydrocarbon_sum);

        return totals;
    }

    
    void outputTimestampFIP(SimulatorTimer& timer, const std::string version)
    {   
        std::ostringstream ss;
        boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%d %b %Y");
        ss.imbue(std::locale(std::locale::classic(), facet));
        ss << "\n                              **************************************************************************\n"
        << "  Balance  at" << std::setw(10) << (double)unit::convert::to(timer.simulationTimeElapsed(), unit::day) << "  Days"
        << " *" << std::setw(30) << eclState().getTitle() << "                                          *\n"
        << "  Report " << std::setw(4) << timer.reportStepNum() << "    " << timer.currentDateTime()
        << "  *                                             Flow  version " << std::setw(11) << version << "  *\n"
        << "                              **************************************************************************\n";
        OpmLog::note(ss.str());
    }


    void outputFluidInPlace(const std::vector<double>& oip, const std::vector<double>& cip, const UnitSystem& units, const int reg)
    {
        std::ostringstream ss;
        if (!reg) {
            ss << "                                                  ===================================================\n"
               << "                                                  :                   Field Totals                  :\n";
        } else {
            ss << "                                                  ===================================================\n"
               << "                                                  :        FIPNUM report region  "
               << std::setw(2) << reg << "                 :\n";
        }
        if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_METRIC) {
            ss << "                                                  :      PAV  =" << std::setw(14) << cip[6] << " BARSA                 :\n"
               << std::fixed << std::setprecision(0)
               << "                                                  :      PORV =" << std::setw(14) << cip[5] << "   RM3                 :\n";
            if (!reg) {
                ss << "                                                  : Pressure is weighted by hydrocarbon pore volume :\n"
                   << "                                                  : Porv volumes are taken at reference conditions  :\n";
            }
            ss << "                         :--------------- Oil    SM3 ---------------:-- Wat    SM3 --:--------------- Gas    SM3 ---------------:\n";
        }
        if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_FIELD) {
            ss << "                                                  :      PAV  =" << std::setw(14) << cip[6] << "  PSIA                 :\n"
               << std::fixed << std::setprecision(0)
               << "                                                  :      PORV =" << std::setw(14) << cip[5] << "   RB                  :\n";
            if (!reg) {
                ss << "                                                  : Pressure is weighted by hydrocarbon pore volume :\n"
                   << "                                                  : Pore volumes are taken at reference conditions  :\n";
            }
            ss << "                         :--------------- Oil    STB ---------------:-- Wat    STB --:--------------- Gas   MSCF ---------------:\n";
        }
        ss << "                         :      Liquid        Vapour        Total   :      Total     :      Free        Dissolved       Total   :" << "\n"
           << ":------------------------:------------------------------------------:----------------:------------------------------------------:" << "\n"
           << ":Currently   in place    :" << std::setw(14) << cip[1] << std::setw(14) << cip[4] << std::setw(14) << (cip[1]+cip[4]) << ":"
           << std::setw(13) << cip[0] << "   :" << std::setw(14) << (cip[2]) << std::setw(14) << cip[3] << std::setw(14) << (cip[2] + cip[3]) << ":\n"
           << ":------------------------:------------------------------------------:----------------:------------------------------------------:\n"
           << ":Originally  in place    :" << std::setw(14) << oip[1] << std::setw(14) << oip[4] << std::setw(14) << (oip[1]+oip[4]) << ":"
           << std::setw(13) << oip[0] << "   :" << std::setw(14) << oip[2] << std::setw(14) << oip[3] << std::setw(14) << (oip[2] + oip[3]) << ":\n"
           << ":========================:==========================================:================:==========================================:\n";
        OpmLog::note(ss.str());
    }


    const EclipseState& eclState() const
    { return ebosSimulator_.gridManager().eclState(); }

    void extractLegacyCellPvtRegionIndex_()
    {
        const auto& grid = ebosSimulator_.gridManager().grid();
        const auto& eclProblem = ebosSimulator_.problem();
        const unsigned numCells = grid.size(/*codim=*/0);

        legacyCellPvtRegionIdx_.resize(numCells);
        for (unsigned cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            legacyCellPvtRegionIdx_[cellIdx] =
                eclProblem.pvtRegionIndex(cellIdx);
        }
    }

    void initHysteresisParams(ReservoirState& state) {
        const int num_cells = Opm::UgGridHelpers::numCells(grid());

        typedef std::vector<double> VectorType;

        const VectorType& somax = state.getCellData( "SOMAX" );

        for (int cellIdx = 0; cellIdx < num_cells; ++cellIdx) {
            ebosSimulator_.model().setMaxOilSaturation(somax[cellIdx], cellIdx);
        }

        if (ebosSimulator_.problem().materialLawManager()->enableHysteresis()) {
            auto matLawManager = ebosSimulator_.problem().materialLawManager();

            VectorType& pcSwMdc_ow = state.getCellData( "PCSWMDC_OW" );
            VectorType& krnSwMdc_ow = state.getCellData( "KRNSWMDC_OW" );

            VectorType& pcSwMdc_go = state.getCellData( "PCSWMDC_GO" );
            VectorType& krnSwMdc_go = state.getCellData( "KRNSWMDC_GO" );

            for (int cellIdx = 0; cellIdx < num_cells; ++cellIdx) {
                matLawManager->setOilWaterHysteresisParams(
                        pcSwMdc_ow[cellIdx],
                        krnSwMdc_ow[cellIdx],
                        cellIdx);
                matLawManager->setGasOilHysteresisParams(
                        pcSwMdc_go[cellIdx],
                        krnSwMdc_go[cellIdx],
                        cellIdx);
            }
        }
    }

    void extractLegacyPoreVolume_()
    {
        const auto& grid = ebosSimulator_.gridManager().grid();
        const unsigned numCells = grid.size(/*codim=*/0);
        const auto& ebosProblem = ebosSimulator_.problem();
        const auto& ebosModel = ebosSimulator_.model();

        legacyPoreVolume_.resize(numCells);
        for (unsigned cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            // todo (?): respect rock compressibility
            legacyPoreVolume_[cellIdx] =
                ebosModel.dofTotalVolume(cellIdx)
                *ebosProblem.porosity(cellIdx);
        }
    }

    void extractLegacyDepth_()
    {
        const auto& grid = ebosSimulator_.gridManager().grid();
        const unsigned numCells = grid.size(/*codim=*/0);

        legacyDepth_.resize(numCells);
        for (unsigned cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            legacyDepth_[cellIdx] =
                grid.cellCenterDepth(cellIdx);
        }
    }

    // Data.
    Simulator& ebosSimulator_;

    std::vector<int> legacyCellPvtRegionIdx_;
    std::vector<double> legacyPoreVolume_;
    std::vector<double> legacyDepth_;
    typedef RateConverter::SurfaceToReservoirVoidage<FluidSystem, std::vector<int> > RateConverterType;
    typedef typename Solver::SolverParameters SolverParameters;

    SimulatorReport failureReport_;

    const ParameterGroup param_;
    ModelParameters model_param_;
    SolverParameters solver_param_;

    // Observed objects.
    BlackoilPropsAdFromDeck& props_;
    NewtonIterationBlackoilInterface& solver_;
    // Misc. data
    const bool has_disgas_;
    const bool has_vapoil_;
    bool       terminal_output_;
    // output_writer
    OutputWriter& output_writer_;
    std::unique_ptr<RateConverterType> rateConverter_;
    // The names of wells that should be defunct
    // (e.g. in a parallel run when they are handeled by
    // a different process)
    std::unordered_set<std::string> defunct_well_names_;

    // Whether this a parallel simulation or not
    bool is_parallel_run_;

};

} // namespace Opm

#endif // OPM_SIMULATORFULLYIMPLICITBLACKOIL_HEADER_INCLUDED
