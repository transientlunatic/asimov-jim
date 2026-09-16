"""Tests for the Jim (JimGW) pipeline integration."""

import os
from unittest.mock import Mock, patch

import pytest
import tomllib
from asimov.pipeline import PipelineException
from asimov.scheduler import Slurm

from asimov_jim import Jim


class TestJimInit:
    """Test Jim initialization."""

    def test_init_success(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        assert pipeline.name == "Jim"
        assert pipeline.production == mock_production
        assert "wait" in pipeline.STATUS

    def test_init_wrong_pipeline(self, mock_production, mock_config):
        mock_production.pipeline = "bilby"
        with pytest.raises(PipelineException, match="Pipeline mismatch"):
            Jim(mock_production)

    def test_init_sets_up_logger(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        assert pipeline.logger is not None


class TestConfigTemplate:
    """Test the config_template property used by asimov's `manage build`
    to render a TOML config when one doesn't already exist in the event
    repository."""

    def test_config_template_is_a_real_bundled_file(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        assert os.path.exists(pipeline.config_template)

    def test_config_template_is_named_jim_toml(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        assert os.path.basename(pipeline.config_template) == "jim.toml"


class TestBuildConfig:
    """Test Jim._build_config(), the Python-side TOML assembly used by
    before_config(). Assembled in Python (not the Liquid template, unlike
    the sibling asimov-pycbc/pycbc.ini) because TOML is type-sensitive
    about strings/numbers/booleans in a way plain text substitution
    handles badly.
    """

    def test_defaults(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["data"]["type"] == "file"
        assert cfg["data"]["detectors"] == ["H1", "L1"]
        assert cfg["data"]["trigger_time"] == 1126259462.4
        assert cfg["waveform"]["approximant"] == "IMRPhenomXPHM"
        assert cfg["waveform"]["f_ref"] == 20
        # Network f_min/f_max: lowest/highest across interferometers.
        assert cfg["likelihood"]["f_min"] == 20
        assert cfg["likelihood"]["f_max"] == 896
        assert cfg["sampler"]["type"] == "flowmc"
        assert cfg["output"]["overwrite"] is True

    def test_data_reads_generic_data_files_block(self, mock_production, mock_config):
        # Always Jim's "file" mode, reading already-resolved frame files
        # from the same generic data: {"data files": {...}} key every
        # pipeline reads -- populated upstream by a data-fetching
        # production (e.g. asimov-gwdata) wired up via `needs:`, kept
        # consistent across pipelines rather than Jim fetching its own
        # data via its built-in gwosc/injection modes.
        mock_production.meta["data"] = {
            "data files": {"H1": "/data/h1.gwf", "L1": "/data/l1.gwf"},
            "channels": {"H1": "H1:GDS-CALIB_STRAIN", "L1": "L1:GDS-CALIB_STRAIN"},
            "segment length": 8.0,
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["data"]["type"] == "file"
        assert cfg["data"]["strain_files"]["H1"] == "/data/h1.gwf"
        assert cfg["data"]["strain_channels"]["L1"] == "L1:GDS-CALIB_STRAIN"
        assert cfg["data"]["duration"] == 8.0

    def test_data_psd_files_is_jim_specific(self, mock_production, mock_config):
        # No generic Asimov vocabulary for PSD files, so these stay under
        # jim: data:.
        mock_production.meta["jim"] = {
            "data": {
                "psd_files": {"H1": "/data/h1_psd.npz", "L1": "/data/l1_psd.npz"},
            }
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["data"]["psd_files"]["L1"] == "/data/l1_psd.npz"

    def test_data_psd_is_asd(self, mock_production, mock_config):
        mock_production.meta["jim"] = {
            "data": {
                "psd_files": {"H1": "/data/h1_asd.npz"},
                "psd_is_asd": {"H1": True},
            }
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["data"]["psd_is_asd"] == {"H1": True}

    def test_data_without_channels_omits_strain_channels(
        self, mock_production, mock_config
    ):
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert "strain_channels" not in cfg["data"]

    def test_output_preserves_user_fields(self, mock_production, mock_config):
        mock_production.meta["jim"] = {
            "output": {"corner_parameters": ["M_c", "q"], "n_samples": 2000},
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["output"]["corner_parameters"] == ["M_c", "q"]
        assert cfg["output"]["n_samples"] == 2000
        # plugin-owned fields are still enforced regardless of user input
        assert cfg["output"]["overwrite"] is True

    def test_sampling_section_passed_through(self, mock_production, mock_config):
        mock_production.meta["jim"] = {
            "sampling": {"time_frame": "geocentric", "sky_frame": "equatorial"},
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["sampling"] == {"time_frame": "geocentric", "sky_frame": "equatorial"}

    def test_sampling_section_omitted_when_not_set(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert "sampling" not in cfg

    def test_jim_seed_override(self, mock_production, mock_config):
        mock_production.meta["jim"] = {"seed": 42}
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["seed"] == 42

    def test_waveform_reads_generic_block_only(self, mock_production, mock_config):
        # Not jim-specific: the same waveform: block every pipeline reads,
        # no jim: override layer.
        mock_production.meta["waveform"] = {
            "approximant": "IMRPhenomXAS",
            "reference frequency": 50.0,
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["waveform"]["approximant"] == "IMRPhenomXAS"
        assert cfg["waveform"]["f_ref"] == 50.0

    def test_likelihood_reads_generic_block_only(self, mock_production, mock_config):
        mock_production.meta["likelihood"] = {
            "minimum frequency": {"H1": 15.0, "L1": 15.0},
            "maximum frequency": {"H1": 512.0, "L1": 512.0},
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["likelihood"]["f_min"] == 15.0
        assert cfg["likelihood"]["f_max"] == 512.0

    def test_sampler_reads_generic_sampler_kwargs_shape(self, mock_production, mock_config):
        # Same generic sampler: {sampler: <name>, "sampler kwargs": {...}}
        # shape bilby/pycbc use.
        mock_production.meta["sampler"] = {
            "sampler": "blackjax-ns-aw",
            "sampler kwargs": {"n_chains": 500},
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["sampler"]["type"] == "blackjax-ns-aw"
        assert cfg["sampler"]["n_chains"] == 500

    def test_sampler_defaults_to_flowmc(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["sampler"]["type"] == "flowmc"

    def test_prior_falls_back_to_prior_interface(self, mock_production, mock_config):
        mock_production.priors = {
            "M_c": {"type": "uniform", "minimum": 10.0, "maximum": 80.0},
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["prior"] == {"M_c": {"type": "uniform", "min": 10.0, "max": 80.0}}

    def test_explicit_jim_prior_overrides_interface(self, mock_production, mock_config):
        mock_production.priors = {
            "M_c": {"type": "uniform", "minimum": 10.0, "maximum": 80.0},
        }
        mock_production.meta["jim"] = {
            "prior": {"M_c": {"type": "uniform", "min": 5.0, "max": 100.0}},
        }
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["prior"] == {"M_c": {"type": "uniform", "min": 5.0, "max": 100.0}}

    def test_checkpoint_dir_distinct_from_output_dir(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["sampler"]["checkpoint_dir"] != cfg["output"]["dir"]

    def test_no_likelihood_frequency_metadata_uses_jim_defaults(
        self, mock_production, mock_config
    ):
        mock_production.meta["likelihood"] = {}
        pipeline = Jim(mock_production)
        cfg = pipeline._build_config()
        assert cfg["likelihood"]["f_min"] == 20.0
        assert cfg["likelihood"]["f_max"] == 1024.0


class TestBeforeConfig:
    def test_before_config_stashes_valid_toml(self, mock_production, mock_config):
        pipeline = Jim(mock_production)
        pipeline.before_config()
        rendered = mock_production.meta["jim_rendered_toml"]
        parsed = tomllib.loads(rendered)
        assert parsed["data"]["type"] == "file"
        assert parsed["output"]["overwrite"] is True

    def test_before_config_resolves_rundir_when_unset(
        self, mock_production, mock_config, temp_dir
    ):
        # before_config() runs before build_dag() (see asimov/cli/manage.py),
        # so production.rundir may still be None here -- regression test for
        # _build_config() crashing with TypeError from os.path.join(None, ...).
        mock_production.rundir = None
        mock_config.get = lambda section, key: (
            temp_dir if (section, key) == ("general", "rundir_default") else ""
        )
        pipeline = Jim(mock_production)
        pipeline.before_config()
        assert mock_production.rundir == os.path.join(
            temp_dir, mock_production.event.name, mock_production.name
        )


class TestBuildDag:
    """Test resolution of rundir and config file location."""

    def test_build_dag_resolves_rundir(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        pipeline = Jim(mock_production)
        toml_file = pipeline.build_dag()
        assert os.path.isdir(mock_production.rundir)
        assert toml_file.endswith("TestProduction.toml")

    def test_build_dag_falls_back_to_rundir_default(
        self, mock_production, mock_config, temp_dir
    ):
        mock_production.rundir = None
        mock_config.get = lambda section, key: (
            temp_dir if (section, key) == ("general", "rundir_default") else ""
        )
        pipeline = Jim(mock_production)
        pipeline.build_dag()
        assert mock_production.rundir == os.path.join(
            temp_dir, mock_production.event.name, mock_production.name
        )

    def test_build_dag_raises_if_no_config_found(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.event.repository.find_prods.return_value = []
        pipeline = Jim(mock_production)
        with pytest.raises(PipelineException, match="No configuration file found"):
            pipeline.build_dag()

    def test_build_dag_dryrun_does_not_raise(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        pipeline = Jim(mock_production)
        pipeline.build_dag(dryrun=True)  # must not raise

    def test_build_dag_without_event_repository(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.event.repository = None
        pipeline = Jim(mock_production)
        toml_file = pipeline.build_dag()
        assert toml_file == "TestProduction.toml"


class TestSubmitDag:
    """Test job submission via Asimov's scheduler abstraction."""

    def test_submit_dag_uses_scheduler_abstraction(
        self, mock_production, mock_config, temp_dir
    ):
        mock_production.rundir = os.path.join(temp_dir, "run")
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()
            pipeline._scheduler.submit.return_value = 12345

            cluster_id = pipeline.submit_dag(dryrun=False)

        assert cluster_id == 12345
        assert mock_production.job_id == 12345
        assert mock_production.status == "running"
        assert pipeline._scheduler.submit.called

    def test_submit_dag_no_gpu_request_by_default(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()
            pipeline._scheduler.submit.return_value = 1

            pipeline.submit_dag(dryrun=False)

        job = pipeline._scheduler.submit.call_args[0][0]
        assert "request_gpus" not in job.kwargs
        assert "slurm_gres" not in job.kwargs

    def test_submit_dag_requests_gpu_on_htcondor(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.meta["scheduler"]["gpus"] = 1
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()  # not a Slurm instance
            pipeline._scheduler.submit.return_value = 1

            pipeline.submit_dag(dryrun=False)

        job = pipeline._scheduler.submit.call_args[0][0]
        assert job.kwargs["request_gpus"] == "1"
        assert "slurm_gres" not in job.kwargs

    def test_submit_dag_requests_gpu_on_slurm(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.meta["scheduler"]["gpus"] = 1
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock(spec=Slurm)
            pipeline._scheduler.submit.return_value = 1

            pipeline.submit_dag(dryrun=False)

        job = pipeline._scheduler.submit.call_args[0][0]
        assert job.kwargs["slurm_gres"] == "gpu:1"
        assert "request_gpus" not in job.kwargs

    def test_submit_dag_dryrun_does_not_call_scheduler(
        self, mock_production, mock_config, temp_dir
    ):
        mock_production.rundir = os.path.join(temp_dir, "run")
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()

            result = pipeline.submit_dag(dryrun=True)

        assert result is None
        pipeline._scheduler.submit.assert_not_called()

    def test_submit_dag_missing_executable_raises_clear_exception(
        self, mock_production, mock_config, temp_dir
    ):
        mock_production.rundir = os.path.join(temp_dir, "run")
        with patch("asimov_jim.pipeline.shutil.which", return_value=None):
            pipeline = Jim(mock_production)
            with pytest.raises(PipelineException, match="jim-run"):
                pipeline.submit_dag(dryrun=False)

    def test_submit_dag_scheduler_failure(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()
            pipeline._scheduler.submit.side_effect = RuntimeError("could not submit")

            with pytest.raises(PipelineException, match="could not be submitted"):
                pipeline.submit_dag(dryrun=False)

    def test_submit_dag_appends_verbose_flag(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.meta["jim"] = {"verbose": True}
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()
            pipeline._scheduler.submit.return_value = 1

            pipeline.submit_dag(dryrun=False)

        job = pipeline._scheduler.submit.call_args[0][0]
        assert "--verbose" in job.kwargs["arguments"]

    def test_submit_dag_sets_accounting_group(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()
            pipeline._scheduler.submit.return_value = 1

            pipeline.submit_dag(dryrun=False)

        job = pipeline._scheduler.submit.call_args[0][0]
        assert job.kwargs["accounting_group"] == "ligo.dev.o4.cbc.pe.jim"


class TestDetectCompletion:
    """Test completion detection via config.final.toml."""

    def test_detect_completion_no_output(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        pipeline = Jim(mock_production)
        assert pipeline.detect_completion() is False

    def test_detect_completion_marker_present(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "config.final.toml"), "w") as f:
            f.write("seed = 0\n")
        pipeline = Jim(mock_production)
        assert pipeline.detect_completion() is True

    def test_detect_completion_ignores_partial_output(self, mock_production, mock_config, temp_dir):
        # samples.npz alone (without config.final.toml) means the run hasn't
        # finished writing outputs yet.
        mock_production.rundir = temp_dir
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "samples.npz"), "w") as f:
            f.write("x")
        pipeline = Jim(mock_production)
        assert pipeline.detect_completion() is False


class TestSamplesAndAssets:
    def test_samples_empty_when_no_output(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        pipeline = Jim(mock_production)
        assert pipeline.samples() == []

    def test_samples_returns_samples_file(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir)
        samples_path = os.path.join(output_dir, "samples.npz")
        with open(samples_path, "w") as f:
            f.write("x")
        pipeline = Jim(mock_production)
        assert pipeline.samples() == [samples_path]

    def test_collect_assets_includes_config(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        pipeline = Jim(mock_production)
        assets = pipeline.collect_assets()
        assert assets["config"] == "TestProduction.toml"
        assert assets["samples"] == []


class TestCollectLogs:
    def test_collect_logs_reads_log_files(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        with open(os.path.join(temp_dir, "TestProduction.out"), "w") as f:
            f.write("stdout contents")
        with open(os.path.join(temp_dir, "TestProduction.err"), "w") as f:
            f.write("stderr contents")
        pipeline = Jim(mock_production)
        logs = pipeline.collect_logs()
        assert logs["TestProduction.out"] == "stdout contents"
        assert logs["TestProduction.err"] == "stderr contents"

    def test_collect_logs_empty_when_no_logs(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = temp_dir
        pipeline = Jim(mock_production)
        assert pipeline.collect_logs() == {}


class TestAfterCompletion:
    def test_after_completion_marks_production_finished(
        self, mock_production, mock_config, temp_dir
    ):
        mock_production.rundir = temp_dir
        mock_production.status = "running"
        pipeline = Jim(mock_production)

        pipeline.after_completion()

        assert mock_production.status == "finished"


class TestResurrect:
    def test_resurrect_resubmits_job(self, mock_production, mock_config, temp_dir):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.meta.pop("resurrections", None)
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()
            pipeline._scheduler.submit.return_value = 99

            pipeline.resurrect()

        assert mock_production.meta["resurrections"] == 1
        assert pipeline._scheduler.submit.called

    def test_resurrect_gives_up_after_five_attempts(
        self, mock_production, mock_config, temp_dir
    ):
        mock_production.rundir = os.path.join(temp_dir, "run")
        mock_production.meta["resurrections"] = 5
        with patch("asimov_jim.pipeline.shutil.which", return_value="/opt/conda/bin/jim-run"):
            pipeline = Jim(mock_production)
            pipeline._scheduler = Mock()

            pipeline.resurrect()

        pipeline._scheduler.submit.assert_not_called()
        assert mock_production.meta["resurrections"] == 5
