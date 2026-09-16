"""Jim (JimGW) pipeline specification for Asimov."""

import glob
import importlib.resources
import os
import shutil

import tomli_w

from asimov import config
from asimov.pipeline import Pipeline, PipelineException
from asimov.scheduler import Slurm
from asimov.scheduler_utils import create_job_from_dict
from asimov.utils import set_directory


class Jim(Pipeline):
    """
    The Jim (JimGW) parameter estimation pipeline integration for Asimov.

    Jim (https://github.com/GW-JAX-Team/Jim) is a JAX-based toolkit for
    Bayesian parameter estimation of gravitational-wave sources. Its
    ``jim-run`` CLI takes a single TOML configuration file and runs to
    completion (or to its next checkpoint) as one job -- there is no
    separate DAG-building step, so this pipeline is submitted directly as
    a single job, following the same pattern used by the sibling
    asimov-pycbc plugin for ``pycbc_inference``.

    Parameters
    ----------
    production : :class:`asimov.Production`
        The production object.
    category : str, optional
        The category of the job. Defaults to "analyses".
    """

    name = "Jim"
    STATUS = {"wait", "stuck", "stopped", "running", "finished"}

    def __init__(self, production, category=None):
        super(Jim, self).__init__(production, category)
        if not production.pipeline.lower() == "jim":
            raise PipelineException("Pipeline mismatch")

    def before_config(self, dryrun=False):
        """
        Build Jim's TOML configuration and stash it as rendered text.

        Asimov's ``manage build`` CLI calls ``production.pipeline.before_config()``
        immediately before ``production.make_config()``, so this is where
        the config gets built. It's done here in Python, rather than with
        Liquid template logic (compare the sibling asimov-pycbc/pycbc.ini,
        which builds its ``.ini`` entirely in the template): TOML is
        type-sensitive about strings vs numbers vs booleans in a way plain
        text substitution handles badly (Python's ``True``/``False`` are
        not valid TOML), so the config is assembled as a real dict and
        serialized with ``tomli_w``, and only the resulting text is handed
        to the (near pass-through) bundled Liquid template.

        Resolves ``production.rundir`` first: this hook runs before
        ``build_dag()`` (see ``_resolve_rundir``'s docstring), but
        ``_build_config()`` needs ``rundir`` set to compute the output/
        checkpoint directories.
        """
        self._resolve_rundir()
        rendered = tomli_w.dumps(self._build_config())
        self.production.meta["jim_rendered_toml"] = rendered

    def _build_config(self):
        """
        Assemble the dict that becomes Jim's TOML config, merging (in
        priority order) explicit ``jim: <section>:`` overrides in the
        production's metadata, common Asimov metadata conventions, and
        Jim's own defaults (see ``jimgw.cli._jim._INIT_TEMPLATE``).
        """
        meta = self.production.meta
        jim_meta = meta.get("jim") or {}
        data_meta = jim_meta.get("data") or {}
        waveform_meta = jim_meta.get("waveform") or {}
        likelihood_meta = jim_meta.get("likelihood") or {}
        sampler_meta = dict(jim_meta.get("sampler") or {})
        output_meta = jim_meta.get("output") or {}
        asimov_waveform = meta.get("waveform") or {}
        ifos = meta.get("interferometers") or []

        data_type = data_meta.get("type", "gwosc")
        data = {
            "type": data_type,
            "detectors": list(ifos),
            "trigger_time": meta.get("event time", data_meta.get("trigger_time")),
        }
        if data_type == "injection":
            data.update(
                {
                    "sampling_frequency": data_meta.get("sampling_frequency", 4096.0),
                    "duration": data_meta.get("duration", 4.0),
                    "zero_noise": bool(data_meta.get("zero_noise", False)),
                    "injection_parameters": dict(
                        data_meta.get("injection_parameters") or {}
                    ),
                }
            )
        elif data_type == "file":
            data.update(
                {
                    "duration": data_meta.get("duration", 4.0),
                    "strain_files": dict(data_meta.get("strain_files") or {}),
                    "psd_files": dict(data_meta.get("psd_files") or {}),
                }
            )
            channels = data_meta.get("strain_channels")
            if channels:
                data["strain_channels"] = dict(channels)
            psd_is_asd = data_meta.get("psd_is_asd")
            if psd_is_asd:
                data["psd_is_asd"] = dict(psd_is_asd)
        else:
            data.update(
                {
                    "duration": data_meta.get("duration", 4.0),
                    "post_trigger_duration": data_meta.get(
                        "post_trigger_duration", 2.0
                    ),
                    "psd_duration": data_meta.get("psd_duration", 1024.0),
                }
            )

        waveform = {
            "approximant": waveform_meta.get(
                "approximant", asimov_waveform.get("approximant", "IMRPhenomXAS")
            ),
            "f_ref": waveform_meta.get(
                "f_ref", asimov_waveform.get("reference frequency", 20.0)
            ),
        }

        prior = jim_meta.get("prior")
        if prior is None:
            prior = self.get_prior_interface().convert()

        asimov_likelihood = meta.get("likelihood") or {}
        f_min_by_ifo = asimov_likelihood.get("minimum frequency") or {}
        f_max_by_ifo = asimov_likelihood.get("maximum frequency") or {}
        # Network min/max frequency: the lowest/highest per-detector value,
        # same convention as the sibling asimov-pycbc/pycbc.ini template --
        # Jim's likelihood takes a single scalar rather than a per-detector one.
        f_min_default = min(f_min_by_ifo.values()) if f_min_by_ifo else 20.0
        f_max_default = max(f_max_by_ifo.values()) if f_max_by_ifo else 1024.0
        likelihood = {
            "f_min": likelihood_meta.get("f_min", f_min_default),
            "f_max": likelihood_meta.get("f_max", f_max_default),
        }

        sampler_meta.setdefault("type", "flowmc")
        sampler_meta["checkpoint_dir"] = self._checkpoint_dir()
        sampler_meta.setdefault("checkpoint_interval", 600.0)

        # Pass through every user-supplied output field (e.g.
        # corner_parameters), then override only the fields this plugin
        # itself owns.
        output = dict(output_meta)
        output.setdefault("n_samples", 5000)
        output.setdefault("save_corner", True)
        output["save_corner"] = bool(output["save_corner"])
        output["dir"] = self._output_dir()
        output["overwrite"] = True

        cfg = {
            "seed": jim_meta.get("seed", 0),
            "data": data,
            "waveform": waveform,
            "prior": prior,
            "likelihood": likelihood,
            "sampler": sampler_meta,
            "output": output,
        }

        sampling = jim_meta.get("sampling")
        if sampling:
            cfg["sampling"] = dict(sampling)

        return cfg

    @property
    def config_template(self):
        """
        The bundled Liquid template used to render a production's TOML
        config file when one doesn't already exist in the event repository.

        Asimov's generic ``manage build`` step calls ``production.make_config()``,
        which looks for this attribute on the pipeline (see
        ``Analysis.make_config`` in asimov core) before falling back to a
        template bundled inside asimov's own package -- which doesn't ship
        a Jim template.
        """
        return str(
            importlib.resources.files("asimov_jim").joinpath("configs/jim.toml")
        )

    def _output_dir(self):
        """
        The directory ``jim-run`` writes ``samples.npz``, ``diagnostics.json``
        and ``config.final.toml`` to (see the bundled ``[output]`` section
        of ``configs/jim.toml``).
        """
        return os.path.join(self.production.rundir, "output")

    def _checkpoint_dir(self):
        """
        The directory ``jim-run`` writes ``checkpoint.pkl`` to (see the
        bundled ``[sampler]`` section of ``configs/jim.toml``). Kept
        separate from ``_output_dir()`` so that jim's own refusal to
        overwrite an existing ``output.dir`` (unless ``output.overwrite``
        is set) isn't tripped by the checkpoint file it writes there while
        still running.
        """
        return os.path.join(self.production.rundir, "checkpoint")

    def _completion_marker(self):
        """
        ``config.final.toml`` is the last file ``write_outputs()`` writes
        in jim-run before the (optional) corner plot, so its presence is a
        reliable signal that a run completed successfully.
        """
        return os.path.join(self._output_dir(), "config.final.toml")

    def _samples_file(self):
        return os.path.join(self._output_dir(), "samples.npz")

    def _executable(self):
        """
        Resolve the ``jim-run`` executable.

        Resolved defensively rather than assuming
        ``config["pipelines"]["environment"]/bin/jim-run`` exists (matching
        the approach used by the sibling asimov-pycbc and asimov-bayeswave
        plugins): in minimal/containerised environments that config value
        may not point at the active environment. ``shutil.which`` also
        picks up an explicit per-production override via
        ``production.meta["executable"]``.
        """
        default_executable = os.path.join(
            config.get("pipelines", "environment"), "bin", "jim-run"
        )
        executable = self.production.meta.get("executable", default_executable)
        executable = shutil.which(executable) or shutil.which("jim-run")
        if executable is None:
            raise PipelineException(
                "Cannot find the jim-run executable",
                production=self.production.name,
            )
        return executable

    def _resolve_rundir(self):
        """
        Resolve (and create) ``self.production.rundir``, shared by
        ``before_config`` and ``build_dag``.

        ``before_config()`` runs before ``build_dag()`` (see the call sites
        in ``asimov/cli/manage.py``), so ``production.rundir`` may still be
        unset at that point; ``_build_config()`` needs it (via
        ``_output_dir``/``_checkpoint_dir``) to render the TOML, so both
        entry points resolve it the same way rather than only ``build_dag``
        doing so and leaving ``before_config`` to crash on ``None``.
        """
        if self.production.rundir:
            self.production.rundir = os.path.abspath(self.production.rundir)
        else:
            self.production.rundir = os.path.join(
                config.get("general", "rundir_default"),
                self.production.event.name,
                self.production.name,
            )
        os.makedirs(self.production.rundir, exist_ok=True)
        return self.production.rundir

    def build_dag(self, dryrun=False):
        """
        Resolve the run directory and the location of the rendered TOML
        config file for this production.

        ``jim-run`` has no separate DAG-building step of its own -- it is
        submitted directly as a single job (see ``submit_dag``) -- but
        asimov's generic ``manage build submit`` CLI unconditionally calls
        ``build_dag`` on every pipeline before ``submit_dag``, so this must
        exist.
        """
        self._resolve_rundir()

        if self.production.event.repository:
            configs = self.production.event.repository.find_prods(
                self.production.name, self.category
            )
            if not configs:
                raise PipelineException(
                    f"No configuration file found for {self.production.name} "
                    f"in the event repository's '{self.category}' directory.",
                    production=self.production.name,
                )
            toml_file = os.path.join(
                self.production.event.repository.directory, self.category, configs[0]
            )
        else:
            toml_file = f"{self.production.name}.toml"

        if dryrun:
            print(f"Configuration file: {toml_file}")

        return toml_file

    def submit_dag(self, dryrun=False):
        """
        Submit a ``jim-run`` job to the scheduler.

        Uses Asimov's scheduler abstraction (``self.scheduler``, HTCondor
        or Slurm) rather than hand-rolling ``htcondor``/``htcondor2`` calls
        directly. Jim is JAX-based and typically wants a GPU: set
        ``scheduler: {gpus: N}`` in the production's metadata to request
        one (``request_gpus`` under HTCondor, ``--gres=gpu:N`` under Slurm).

        Parameters
        ----------
        dryrun : bool, optional
            If True then the job will not be submitted, but the command and
            submit description will be printed to stdout.

        Returns
        -------
        int
            The cluster ID assigned to the submitted job.

        Raises
        ------
        PipelineException
            This will be raised if the pipeline fails to submit the job.
        """
        toml_file = self.build_dag(dryrun=dryrun)
        rundir = self.production.rundir

        command = [self._executable(), toml_file]
        if self.production.meta.get("jim", {}).get("verbose", False):
            command.append("--verbose")

        scheduler_meta = self.production.meta.get("scheduler", {})
        batch_name = f"Jim/{self.production.event.name}/{self.production.name}"

        self.logger.info(" ".join(command))

        with open(os.path.join(rundir, "jim.sh"), "w") as script_file:
            script_file.write(" ".join(command) + "\n")

        submit_description = {
            "executable": command[0],
            "arguments": " ".join(command[1:]),
            "output": os.path.join(rundir, f"{self.production.name}.out"),
            "error": os.path.join(rundir, f"{self.production.name}.err"),
            "log": os.path.join(rundir, f"{self.production.name}.log"),
            "request_cpus": str(scheduler_meta.get("cpus", 1)),
            "request_memory": str(scheduler_meta.get("request memory", "8192MB")),
            "request_disk": str(scheduler_meta.get("request disk", "4096MB")),
            "getenv": "true",
            "batch_name": batch_name,
        }

        gpus = scheduler_meta.get("gpus", 0)
        if gpus:
            if isinstance(self.scheduler, Slurm):
                submit_description["slurm_gres"] = f"gpu:{gpus}"
            else:
                submit_description["request_gpus"] = str(gpus)

        accounting_group = scheduler_meta.get("accounting group")
        if accounting_group:
            submit_description["accounting_group_user"] = config.get(
                "condor", "user"
            )
            submit_description["accounting_group"] = accounting_group

        if dryrun:
            print(f"Would submit: {' '.join(command)}")
            print("SUBMIT DESCRIPTION")
            print("-------------------")
            print(submit_description)
            return None

        with set_directory(rundir):
            try:
                job = create_job_from_dict(submit_description)
                cluster_id = self.scheduler.submit(job)
            except (FileNotFoundError, RuntimeError) as error:
                raise PipelineException(
                    f"The jim-run job could not be submitted: {error}",
                    production=self.production.name,
                ) from error

        self.production.status = "running"
        self.production.job_id = int(cluster_id)
        self.logger.info(f"Submitted Jim job: {self.production.job_id}")
        return int(cluster_id)

    def detect_completion(self):
        """
        Check for ``config.final.toml``, the last file ``jim-run`` writes
        before its (optional) corner plot, to signal that the job has
        completed.
        """
        return os.path.exists(self._completion_marker())

    def samples(self):
        """
        Collect the samples archive for downstream post-processing.
        """
        samples_file = self._samples_file()
        return [samples_file] if os.path.exists(samples_file) else []

    def collect_assets(self):
        """
        Gather the results assets for this job, so that a downstream
        production (for example a post-processing production wired up via
        ``needs:``) can pick them up through
        ``production._previous_assets()``.
        """
        assets = {"samples": self.samples()}
        if self.production.event.repository:
            configs = self.production.event.repository.find_prods(
                self.production.name, self.category
            )
            if configs:
                assets["config"] = configs[0]
        return assets

    def collect_logs(self):
        """
        Collect all of the log files which have been produced by this
        production and return their contents as a dictionary.
        """
        logs = (
            glob.glob(f"{self.production.rundir}/*.err")
            + glob.glob(f"{self.production.rundir}/*.out")
            + glob.glob(f"{self.production.rundir}/*.log")
        )
        messages = {}
        for log in logs:
            with open(log, "r") as log_f:
                messages[os.path.basename(log)] = log_f.read()
        return messages

    def after_completion(self):
        """
        Mark this production as finished once its job has completed.

        This deliberately does not reach out and submit any post-processing
        job itself. Post-processing is instead expressed as its own,
        separate production with a ``needs:`` dependency on this one --
        see the sibling asimov-pycbc plugin for the same pattern.
        """
        super().after_completion()

    def resurrect(self):
        """
        Attempt to resurrect a failed or evicted job by resubmitting it.

        ``jim-run`` automatically resumes sampling from ``checkpoint.pkl``
        (written every ``sampler.checkpoint_interval`` seconds to
        ``sampler.checkpoint_dir`` -- see ``configs/jim.toml``) whenever
        that file exists, with no special command-line flag needed, so
        resubmitting the exact same command (``submit_dag``) picks up from
        the last checkpoint.
        """
        try:
            count = self.production.meta["resurrections"]
        except KeyError:
            count = 0
        if count < 5:
            count += 1
            self.production.meta["resurrections"] = count
            self.submit_dag()

    def get_prior_interface(self):
        from .priors import JimPriorInterface

        if self._prior_interface is None:
            self._prior_interface = JimPriorInterface(self.production.priors)
        return self._prior_interface
