"""Simulations for (real) time evolution, time dependent correlation
functions and spectral functions."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import warnings

from . import simulation
from .simulation import *  # noqa F403
from ..networks.mps import MPSEnvironment, MPS
from ..networks.mpo import MPO
from ..tools.misc import to_iterable, consistency_check
from ..tools import hdf5_io
from ..linalg import np_conserved as npc

__all__ = simulation.__all__ + [
    'RealTimeEvolution', 'SpectralSimulation', 'TimeDependentCorrelation',
    'TimeDependentCorrelationEvolveBraKet', 'SpectralSimulationEvolveBraKet'
]


class RealTimeEvolution(Simulation):
    """Perform a real-time evolution on a tensor network state.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.

    Options
    -------
    .. cfg:config :: TimeEvolution
        :include: Simulation

        final_time : float
            Mandatory. Perform time evolution until ``engine.evolved_time`` reaches this value.
            Note that we can go (slightly) beyond this time if it is not a multiple of
            the individual time steps.
    """
    default_algorithm = 'TEBDEngine'
    default_measurements = Simulation.default_measurements + [
        ('tenpy.simulations.measurement', 'm_evolved_time'),
    ]

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)
        self.final_time = self.options['final_time'] - 1.e-10  # subtract eps: roundoff errors

    def run_algorithm(self):
        """Run the algorithm.

        Calls ``self.engine.run()`` and :meth:`make_measurements`.
        """
        # TODO: more fine-grained/custom break criteria?
        while True:
            if np.real(self.engine.evolved_time) >= self.final_time:
                break
            self.logger.info("evolve to time %.2f, max chi=%d", self.engine.evolved_time.real,
                             max(self.psi.chi))
            self.engine.run()
            # for time-dependent H (TimeDependentExpMPOEvolution) the engine can re-init the model;
            # use it for the measurements....
            self.model = self.engine.model

            self.make_measurements()
            self.engine.checkpoint.emit(self.engine)  # TODO: is this a good idea?

    def perform_measurements(self):
        if getattr(self.engine, 'time_dependent_H', False):
            # might need to re-initialize model with current time
            # in particular for a sequential/resume run, the first `self.init_model()` might not
            # yet have had the initial start time of the algorithm engine!
            self.engine.reinit_model()
            self.model = self.engine.model
        return super().perform_measurements()

    def resume_run_algorithm(self):
        self.run_algorithm()

    def final_measurements(self):
        """Do nothing.

        We already performed a set of measurements after the evolution in :meth:`run_algorithm`.
        """
        pass


class TimeDependentCorrelation(RealTimeEvolution):
    r"""Specialized :class:`RealTimeEvolution` to calculate a time dependent correlation function of a ground state.

    In general this calculates an overlap of the form :math:`C(r, t) = \langle\psi_0| B_r(t) A_{r_0} |\psi_0\rangle`
    where :math:`r_0` is the translationally invariant center of the model.
    This class assumes that :math:`|\psi_0\rangle` is a ground-state. In order to evolve arbitrary initial states,
    the :class:`TimeDependentCorrelationEvolveBraKet` should be used.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.
        These parameters are converted to a (dict-like) :class:`~tenpy.tools.params.Config`.
        For command line use, a ``.yml`` file should hold the information.
    ground_state_filename: str
        the filename as in :cfg:option:`Simulation.output_filename` from a finished
        :class:`tenpy.simulations.ground_state_search.GroundStateSearch`
    ground_state_data: dict
        the ground-state data as a dictionary, i.e. the `gs_results` when running a
        :class:`tenpy.simulations.ground_state_search.GroundStateSearch`

    Options
    -------
    .. cfg:config :: TimeDependentCorrelation
        :include: TimeEvolution

        mixed_space : bool
            Whether to use the mixed-space operator representation (for 2D-lattices only) or not.
        ground_state_filename : str
            a filename of a given ground state search (ideally a hdf5 file coming from a finished
            run of a :class:`~tenpy.simulations.ground_state_search.GroundStateSearch`)
        operator_t0 : dict
            Mandatory, this must fully specify the operator initially applied to the MPS (i.e. before a time
            evolution). Two different "types" of operators are supported.
            An operator can be specified by an ``opname`` (str) and its corresponding
            index - this can be a ``mps_index`` (int) *or* a ``lat_index`` (list[int]).
            To specify a product operator, the ``opname`` and ``mps_index`` or ``lat_index`` can be passed
            as lists where ``opname[i]`` corresponds to ``index[i]```.

            .. note ::
                The ``lat_idx`` must have a (dim+1) length, i.e. ``[x, y, u]``,
                where ``u = 0`` for a single-site unit cell.

            Furthermore, mixed space operators (when the corresponding config option ``mixed_space`` is set to True)
            can be used. A mixed space operator is an operator :math:`O` with a fixed momentum along the y-direction.
            :math:`\hat{O}_x(k_y) = \frac{1}{\sqrt{L_y}} \sum_y e^{i k_y y} \hat{O}_{x + y}`.
            In this "mode" a single ``opname`` is necessary. A definite momentum ``k_y`` (defaults to 0) can
            be specified - note that the range of allowed momenta is restricted based on the "width" of the cylinder.
            No index can be specified as the mixed space operator is applied at the middle of the cylinder in
            x-direction and at y=0. Only the default mps-winding and only single site unit cell models are supported.

            .. warning ::
                This is an experimental feature and probably does not yet work with all
                classes in :mod:`time_evolution`.

            The optional ``key_name`` can specify the result in the measurements after the preceded string
            ``'correlation_function_t_'`` and the key for the time evolved operator in either mode.
        operator_t : dict
            Optional. The setup is mostly identical to the configuration for operator_t0.
            If this is left out, the hermitian conjugate of the `operator_t0` will be used.
            If a product operator is specified, the indices can still be used to specify the *relative* "form"
            of the operator - however, they will not be absolute, as this operator is applied starting at
            MPS index 0 and moving through the whole MPS chain. This works because the operator_t is converted
            into a list of onsite operators (starting at the first site the operator acts on and ending at the
            last site), together with the integer this list of operators is first applied at - this integer is then
            dropped.
        correlation_function_key : str
            This manually sets the name for the correlation function and overrides the ``key_name``
            In the operator config.
    """
    default_measurements = RealTimeEvolution.default_measurements + [
        ('simulation_method', 'm_correlation_function'),
    ]

    def __init__(self, options, *, ground_state_data=None, ground_state_filename=None, **kwargs):
        super().__init__(options, **kwargs)

        resume_data: dict | None = kwargs.get("resume_data", None)
        if resume_data is not None:
            if 'psi_ground_state' in resume_data:
                self.psi_ground_state = resume_data['psi_ground_state']
            else:
                self.logger.warning("psi_ground_state not in resume data")
            if 'gs_energy' in resume_data:
                self.gs_energy = resume_data['gs_energy']
            else:
                self.logger.warning("ground-state energy not in resume data")

        if not self.loaded_from_checkpoint:
            ground_state_data = self._get_ground_state_data(ground_state_filename, ground_state_data)
            if ground_state_data is not None:
                self.logger.info("Initializing from ground state data")
                self._init_from_gs_data(ground_state_data)

        # will be read out in init_state
        self.gs_energy = self.options.get('gs_energy', None)
        # generate info for operator before time evolution as subconfig
        self.operator_t0_config = self.options.subconfig('operator_t0')
        self.operator_t_config = self.options.subconfig('operator_t')
        self.operator_t0: list | None = None
        self.operator_t: tuple | None = None
        # read out config after model initialization in init_state, since most defaults depend on model params
        self.correlation_function_key = self.options.get('correlation_function_key', None)
        self.mixed_space = self.options.get('mixed_space', False)  # TODO: Make work with other classes

    def resume_run(self):
        if not hasattr(self, 'psi_ground_state'):
            # didn't get psi_ground_state in resume_data, but might still have it in the results
            if 'psi_ground_state' not in self.results:
                raise ValueError("psi_ground_state not saved in checkpoint results: can't resume!")
        super().resume_run()

    def get_resume_data(self):
        resume_data = super().get_resume_data()
        resume_data['psi_ground_state'] = self.psi_ground_state
        resume_data['gs_energy'] = self.gs_energy
        return resume_data

    def _get_ground_state_data(self, ground_state_filename, ground_state_data):
        """Get data for the groundstate either from simulation kwargs or entry in options"""
        if ground_state_filename is None:
            ground_state_filename = self.options.get('ground_state_filename', None)
        if ground_state_data is None and ground_state_filename is not None:
            self.logger.info(
                f"loading data from 'ground_state_filename'='{ground_state_filename}'")
            ground_state_data = hdf5_io.load(ground_state_filename)
        elif ground_state_data is not None and ground_state_filename is not None:
            self.logger.warning(
                "Supplied a 'ground_state_filename' and ground_state_data as kwarg. "
                "Ignoring 'ground_state_filename'.")
        return ground_state_data

    def init_measurements(self):
        use_default_meas = self.options.silent_get('use_default_measurements', True)
        connect_any_meas = self.options.silent_get('connect_measurements', None)
        if use_default_meas is False and connect_any_meas is None:
            warnings.warn(f"No measurements are being made, this might not make sense for {self.__class__}")
        super().init_measurements()

    def init_state(self):
        # make sure state is not reinitialized if psi and psi_ground_state are given
        if not hasattr(self, 'psi_ground_state'):
            warnings.warn(
                f"No ground state data is supplied, calling the initial state builder on "
                f"{self.__class__.__name__} class - you probably want to supply a ground state!")
            super().init_state()  # this sets self.psi from init_state_builder (should be avoided)
            self.psi_ground_state = self.psi.copy()
            delattr(self, 'psi')  # free memory

        # configure states here
        self._configure_operator_t0()
        self._configure_operator_t()
        # set keyname
        if self.correlation_function_key is None:
            if self.operator_t_name is None and self.operator_t_name is None:
                self.correlation_function_key = "correlation_function_t"
            elif self.operator_t_name is None:
                self.correlation_function_key = f"correlation_function_t_?_{self.operator_t0_name}"
            elif self.operator_t0_name is None:
                self.correlation_function_key = f"correlation_function_t_{self.operator_t_name}_?"

        if not hasattr(self, 'psi'):
            # copy is essential, since time evolution is probably only performed on psi
            self.psi = self.psi_ground_state.copy()
            self.apply_operator_t0_to_psi()

        # check for saving
        if self.options.get('save_psi', True):
            self.results['psi'] = self.psi
            self.results['psi_ground_state'] = self.psi_ground_state

    def init_algorithm(self, **kwargs):
        super().init_algorithm(**kwargs)  # links to RealTimeEvolution class, not to Simulation
        # make sure to get the energy of the ground state, this is needed for the correlation_function
        if self.gs_energy is None:
            self.gs_energy = self.model.H_MPO.expectation_value(self.psi_ground_state)
            self.logger.info(f"Calculated Energy of initial state as {self.gs_energy:.5f}")
        if self.engine.psi.bc != 'finite':
            raise NotImplementedError(
                "Only finite MPS boundary conditions are currently implemented for "
                f"{self.__class__.__name__}")

    def _init_from_gs_data(self, gs_data):
        if isinstance(gs_data, MPS):
            # self.psi_ground_state = gs_data ?
            raise NotImplementedError(
                "Only hdf5 and dictionaries are supported as ground state input")
        sim_class = gs_data['version_info']['simulation_class']
        if sim_class != 'GroundStateSearch':
            warnings.warn("The Simulation is not loaded from a GroundStateSearch.")

        data_options = gs_data['simulation_parameters']
        for key in data_options:
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = data_options[key]
            elif self.options[key] != data_options[key]:
                warnings.warn(
                    "Different model parameters in Simulation and data from file. Ignoring parameters "
                    "in data from file")
        if 'energy' in gs_data:
            self.options['gs_energy'] = gs_data['energy']

        if 'psi' not in gs_data:
            raise ValueError("MPS for ground state not found")
        psi_ground_state = gs_data['psi']
        if not isinstance(psi_ground_state, MPS):
            raise TypeError("Ground state must be an MPS class")

        if not hasattr(self, 'psi_ground_state'):
            self.psi_ground_state = psi_ground_state

    def _configure_operator_t0(self):
        key_name = self.operator_t0_config.get('key_name', None)
        if self.mixed_space is True:
            self.operator_t0: list = self._get_mixed_space_operator_from_config(self.operator_t0_config, self.model.lat)
            if key_name is None:
                key_name = to_iterable(self.operator_t0_config['opname'])[0]
        else:
            self.operator_t0: list = self._get_product_operator_from_config(self.operator_t0_config, self.model.lat)
            if key_name is None:
                if len(self.operator_t0) != 1:
                    if self.correlation_function_key is None:
                        warnings.warn("A 'key_name' should be passed for multiple operators if it was not "
                                      "set explicitly as 'correlation_function_key'")
                else:
                    key_name = self.operator_t0[0][0]
        self.operator_t0_name = key_name

    def _configure_operator_t(self):
        if self.mixed_space is True:
            if self.operator_t_config:
                self.operator_t: tuple = tuple(self._get_mixed_space_operator_from_config(
                    self.operator_t_config, self.model.lat))
                key_name = self.operator_t_config.get('key_name', None)
                if key_name is None:
                    key_name = to_iterable(self.operator_t_config['opname'])[0]
            else:
                assert len(self.operator_t0) == 1
                w_tot, _ = self.operator_t0[0]
                Lx = self.model.lat.shape[0]
                x_indices = self.model.lat.lat2mps_idx([[x, 0, 0] for x in range(Lx)])
                self.operator_t: tuple = (w_tot.conj(), x_indices, False)
                # tuple of conjugate operator indices, and need_JW=False
                key_name = None if self.operator_t0_name is None else self.operator_t0_name + '_dagger'
        else:  # get a term list
            if self.operator_t_config:  # empty config would evaluate to False
                term_list = self._get_product_operator_from_config(self.operator_t_config, self.model.lat)
                key_name = self.operator_t_config.get('key_name', None)
                if key_name is None:
                    if len(term_list) != 1:
                        if self.correlation_function_key is None:
                            warnings.warn("A 'key_name' should be passed for multiple operators if it was not "
                                          "set explicitly as 'correlation_function_key'")
                    else:
                        key_name = term_list[0][0]
            else:
                sites = self.model.lat.mps_sites()
                conjugated_ops = []
                indices = []
                for op, idx in self.operator_t0:
                    conjugated_ops.append(sites[idx].get_hc_op_name(op))
                    indices.append(idx)
                term_list = list(zip(conjugated_ops, indices))
                if len(term_list) == 1:
                    key_name = term_list[0][0]
                else:
                    key_name = None if self.operator_t0_name is None else self.operator_t0_name + '_dagger'
            # convert the term list into a list of operators and provide a list on which to apply them
            ops, i_min, has_extra_JW = self.psi_ground_state._term_to_ops_list(term_list)
            indices_list = np.arange(self.model.lat.N_sites - (len(ops) - 1))
            self.operator_t: tuple = (ops, indices_list, has_extra_JW)

        # set the name of the operator
        self.operator_t_name = key_name

    @ staticmethod
    def _get_product_operator_from_config(subconfig, lat) -> list[tuple]:
        ops = to_iterable(subconfig['opname'])  # opname is mandatory
        mps_idx = subconfig.get('mps_idx', None)
        lat_idx = subconfig.get('lat_idx', None)
        if mps_idx is not None and lat_idx is not None:
            raise KeyError("Either 'mps_idx' or a 'lat_idx' should be passed, not both.")

        if mps_idx is not None:
            idx = mps_idx
        elif lat_idx is not None:
            idx = lat.lat2mps_idx(lat_idx)
        else:
            mid = np.array(lat.shape) // 2  # default to the middle of the Lattice
            idx = lat.lat2mps_idx(mid)

        idx = to_iterable(idx)  # make index an iterable for tiling
        # tiling
        if len(ops) > len(idx):
            if len(idx) != 1:
                raise ValueError(
                    "Ill-defined tiling: num. of operators must be equal to num. of indices or one")
            idx = idx * len(ops)
        elif len(ops) < len(idx):
            if len(ops) != 1:
                raise ValueError(
                    "Ill-defined tiling: num. of operators must be equal to num. of indices or one")
            ops = ops * len(idx)
        # generate list of tuples of form [(op1, i_1), (op2, i_2), ...]
        term_list = list(zip(ops, idx))
        return term_list

    @staticmethod
    def _get_mixed_space_operator_from_config(subconfig, lat) -> list[tuple]:
        ops = to_iterable(subconfig['opname'])
        assert len(ops) == 1, "Only mixed space operator along one ring is supported!"
        assert lat.Lu == 1 and lat.dim == 2, "Only 2d lattice with single-site unit cell supported"
        assert np.array_equal(lat.order, lat.ordering('default')), "Only default lattice order supported"
        # get required momentum
        ky = subconfig.get('ky', None)  # TODO: should we include a test if the momentum is valid?
        if ky is None:
            warnings.warn("No momentum 'ky' for mixed space operator is passed, setting default to 0")
            ky = 0
        # y indices (a_y = 1, lattice spacing in y-dir, we excluded 2 site unit cells above)
        ys = np.arange(lat.shape[1])  # Ly is lat.shape[1]
        # coefficients in FT
        coeffs = np.exp(1j * ys * ky)
        # normalize them
        coeffs /= np.linalg.norm(coeffs)  
        sites = []
        mps_sites = lat.mps_sites()
        for y in ys:
            idx = np.array([lat.shape[0]//2, y, 0])
            sites.append(mps_sites[lat.lat2mps_idx(idx)])
        # mixed space 'MPO'
        mixed_space_mpo = MPO.from_wavepacket(sites, coeffs, ops[0])
        for i, op_i in enumerate(mixed_space_mpo._W):
            op_i.ireplace_label('p', f'p{i}')
            op_i.ireplace_label('p*', f'p{i}*')
            if i == 0:
                w_tot = op_i
            else:
                w_tot = npc.tensordot(w_tot, op_i, axes=('wR', 'wL'))
        try:
            w_tot = npc.trace(w_tot, 'wL', 'wR')  # this does not work when there is charge conservation
        except ValueError:
            w_tot = w_tot.squeeze() 
            
        x_ind_0 = lat.lat2mps_idx(np.array([lat.shape[0]//2, 0, 0]))
        op_list = list(zip([w_tot], x_ind_0))
        return op_list

    def apply_operator_t0_to_psi(self):
        self.logger.info("Applying 'operator_t0' to psi")
        ops_term = self.operator_t0
        if len(ops_term) == 1:
            op, i = ops_term[0]
            self.psi.apply_local_op(i, op)
        else:
            self.psi.apply_local_term(ops_term)

    def m_correlation_function(self, results, psi, model, simulation, **kwargs):
        r"""Measurement function for time dependent correlations.

        This calculates the overlap of :math:`\langle\psi| \text{operator}_t |\phi\rangle`,
        where :math:`|\phi\rangle = e^{-iHt} \text{operator}_{t0} |\psi_0\rangle`
        (the time evolved state after operator_t0 was applied at the specified position (defaults to center)) and
        :math:`e^{i E_0 t} \langle\psi|`.
        """
        self.logger.info("calling m_correlation_function")
        results_key = self.correlation_function_key
        psi_gs = self.psi_ground_state
        env = MPSEnvironment(psi_gs, psi)
        phase = np.exp(1j * self.gs_energy * self.engine.evolved_time)
        ops, indices_list, has_extraJW = self.operator_t
        if self.mixed_space:
            w_tot_conj = ops
            results[results_key] = env.expectation_value(w_tot_conj, indices_list) * phase
        else:
            expvals = []
            for idx in indices_list:
                expvals.append(env.expectation_value_multi_sites(ops, idx, insert_JW_from_left=has_extraJW)*phase)
            results[results_key] = np.array(expvals)


class TimeDependentCorrelationEvolveBraKet(TimeDependentCorrelation):
    r"""Evolving the bra and ket state in :class:`TimeDependentCorrelation`.

    This class allows the calculation of a time-dependent correlation function for arbitrary states
    :math:`|\psi\rangle` (not necessarily ground-states).
    The time-dependent correlation function is :math:`C(r, t) = \langle\psi| e^{i H t} B e^{-i H t} A |\psi\rangle`
    where `B` is the ``operator_t`` and `A` is the ``operator_t0`` at the given site.

    .. note ::

        Any (custom) measurement function and default measurement are measuring with respect to the
        state where the ``operator_t0`` was already applied, that is w.r.t. :math:`A |\psi\rangle`


    Options
    -------
    .. cfg:config :: TimeDependentCorrelationEvolveBraKet
        :include: TimeDependentCorrelation

    """

    def __init__(self, *args, **kwargs):
        self.engine_bra = None  # a second engine will be instantiated in :meth:`init_algorithm`
        resume_data: dict | None = kwargs.get('resume_data', None)
        if resume_data is not None:
            if 'resume_data_bra' in resume_data:
                if 'psi' in resume_data['resume_data_bra']:
                    resume_data['psi_ground_state'] = resume_data['resume_data_bra']['psi']
        super().__init__(*args, **kwargs)

    def init_algorithm(self, **kwargs):
        resume_data_bra = None
        if 'resume_data' in self.results:
            if 'resume_data_bra' in self.results['resume_data']:
                self.logger.info("use `resume_data` for initializing the algorithm engine")
                resume_data_bra = self.results['resume_data']['resume_data_bra'].copy()
                # clean up: they are no longer up to date after algorithm initialization!
                # up to date resume_data is added in :meth:`prepare_results_for_save`
                self.results['resume_data']['resume_data_bra'].clear()
                del self.results['resume_data']['resume_data_bra']

        super().init_algorithm(**kwargs)  # links to Simulation
        if resume_data_bra is not None:
            kwargs.setdefault('resume_data', resume_data_bra)
            if 'psi' in resume_data_bra:
                self.psi_ground_state = resume_data_bra['psi']  # make sure to use resume data of bra
        kwargs.setdefault('cache', self.cache)  # TODO: can we use the same cache
        # make sure a second engine is used when evolving the bra
        # fetch engine that evolves ket
        AlgorithmClass = self.engine.__class__
        # instantiate the second engine for the ground state
        algorithm_params = self.options.subconfig('algorithm_params')
        self.engine_bra = AlgorithmClass(self.psi_ground_state, self.model,
                                         algorithm_params, **kwargs)

    def run_algorithm(self):
        while True:
            if np.real(self.engine.evolved_time) >= self.final_time:
                break
            self.logger.info("evolve to time %.2f, max chi=%d", self.engine.evolved_time.real,
                             max(self.psi.chi))
            self.engine_bra.run()  # first evolve bra
            # call engine_bra_resume_data in case something else is done here....
            self.engine.run()  # evolve ket (psi)
            # sanity check, bra and ket should evolve to same time
            assert np.isclose(self.engine_bra.evolved_time, self.engine.evolved_time), ('Bra evolved to different time '
                                                                                        'than ket')
            self.model = self.engine.model
            self.make_measurements()
            self.engine.checkpoint.emit(self.engine)  # set up in init_algorithm of Simulation class
            # and connects to self.save_at_checkpoint which ultimately calls self.save_results()
            # if there are no results self.prepare_results_for_save() which calls get_resume_data() which
            # calls engine.get_resume_data

    def m_correlation_function(self, results, psi, model, simulation, **kwargs):
        """Equivalent to :meth:`TimeDependentCorrelation.m_correlation_function`."""
        self.logger.info("calling m_correlation_function")
        results_key = self.correlation_function_key
        psi_bra = self.engine_bra.psi
        if self.grouped > 1:
            psi_bra = psi_bra.copy()  # make copy since algorithm might use grouped bra
            psi_bra.group_split(self.options['algorithm_params']['trunc_params'])
        env = MPSEnvironment(psi_bra, psi)
        ops, indices_list, has_extraJW = self.operator_t
        if self.mixed_space:
            w_tot_conj = ops
            results[results_key] = env.expectation_value(w_tot_conj, indices_list)
        else:
            expvals = []
            for idx in indices_list:
                expvals.append(env.expectation_value_multi_sites(ops, idx, insert_JW_from_left=has_extraJW))
            results[results_key] = np.array(expvals)

    def get_resume_data(self) -> dict:
        """Get resume data for a Simulation for two engines."""
        resume_data = super(TimeDependentCorrelation, self).get_resume_data()  # call Simulation's method
        # in order not to write the ground-state twice into the resume_data
        resume_data_bra = self.engine_bra.get_resume_data()
        resume_data['resume_data_bra'] = resume_data_bra
        resume_data['gs_energy'] = self.gs_energy
        return resume_data

    def estimate_RAM(self):
        engine_ket_RAM = super().estimate_RAM()
        engine_bra_RAM = self.engine_bra.estimate_RAM()
        return engine_ket_RAM + engine_bra_RAM

    def group_sites_for_algorithm(self):
        super().group_sites_for_algorithm()
        bra = self.psi_ground_state
        if self.grouped > 1:
            if not self.loaded_from_checkpoint or bra.grouped < self.grouped:
                bra.group_sites(self.grouped)

    def group_split(self):
        """Split sites of psi that were grouped in  :meth:`group_sites_for_algorithm`."""
        bra = self.psi_ground_state
        if self.grouped > 1:
            bra.group_split(self.options['algorithm_params']['trunc_params'])
            self.psi.group_split(self.options['algorithm_params']['trunc_params'])
            self.model = self.model_ungrouped
            del self.model_ungrouped
            self.grouped = 1


class SpectralSimulation(TimeDependentCorrelation):
    """Simulation class to calculate Spectral Functions.

    The interface to the class is the same as to :class:`TimeDependentCorrelation`.

    Options
    -------
    .. cfg:config :: SpectralSimulation
        :include: TimeDependentCorrelation

        spectral_function_params : dict
            Additional parameters for post-processing of the spectral function (i.e. applying
            linear prediction or gaussian windowing. The keys correspond to the kwargs of
            :func:`~tenpy.tools.spectral_function_tools.spectral_function`.
        max_rel_prediction_time : float | None
            Threshold for raising errors on using too much linear prediction. Default ``3``.
            See :meth:`~tenpy.tools.misc.consistency_check`.
            Can be downgraded to a warning by setting this option to ``None``.
        results_key : str
            Optional. The key for the processed spectral function.
    """

    def __init__(self, options, *, ground_state_data=None, ground_state_filename=None, **kwargs):
        super().__init__(options,
                         ground_state_data=ground_state_data,
                         ground_state_filename=ground_state_filename,
                         **kwargs)

    def run_post_processing(self):
        extra_kwargs = self.options.get('spectral_function_params', {})
        consistency_check(value=extra_kwargs.get('rel_prediction_time', 1),
                          options=self.options, threshold_key='max_rel_prediction_time',
                          threshold_default=3,
                          msg="Excessive use of linear prediction; ``max_rel_prediction_time`` exceeded")
        if self.correlation_function_key in self.results['measurements']:
            results_key = self.options.get('results_key', 'spectral_function')
            kwargs_dict = {'results_key': results_key, 'correlation_key': self.correlation_function_key}
            kwargs_dict.update(extra_kwargs)  # add parameters for linear prediction etc.
            if self.mixed_space:
                kwargs_dict.update({'mixed_space': True})
            pp_entry = ('tenpy.simulations.post_processing', 'pp_spectral_function',
                        kwargs_dict)
            # create a new list here! (otherwise this is added to all instances within that session)
            self.default_post_processing = self.default_post_processing + [pp_entry]
        return super().run_post_processing()


class SpectralSimulationEvolveBraKet(SpectralSimulation, TimeDependentCorrelationEvolveBraKet):
    pass
