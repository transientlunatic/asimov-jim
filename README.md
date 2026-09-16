# asimov-jim

[Jim (JimGW)](https://github.com/GW-JAX-Team/Jim) pipeline integration for
[Asimov](https://github.com/transientlunatic/asimov).

Jim is a JAX-based toolkit for Bayesian parameter estimation of
gravitational-wave sources, pairing differentiable waveform models from
[ripple](https://github.com/GW-JAX-Team/ripple) with GPU-accelerated
JAX-based samplers (flowMC and several BlackJAX backends). This plugin lets
Asimov build, submit, monitor and collect the results of `jim-run` jobs
alongside other pipelines (PyCBC, Bilby, LALInference, ...) in the same
project.

## Installation

```bash
pip install asimov-jim
```

This also installs `asimov`. You'll separately need Jim itself available in
the environment the job runs in (`pip install JimGW`, or `JimGW[cuda]` for
GPU support) so that `jim-run` is on the `PATH`.

For development:

```bash
git clone https://github.com/transientlunatic/asimov-jim.git
cd asimov-jim
pip install -e ".[test]"
pytest
```

## Usage

Ledger settings are added via blueprint files and `asimov apply`, not by
hand-editing a flat ledger — and settings are inherited hierarchically
(analysis > event/subject > pipeline defaults > global), so event-wide
values like `interferometers`, `"event time"`, `waveform` and
`likelihood` frequencies are set once on the *event* blueprint rather than
repeated on every analysis. A minimal event blueprint:

```yaml
# event.yaml
kind: event
name: GW150914_095045

interferometers: [H1, L1]
"event time": 1126259462.4

waveform:
  approximant: IMRPhenomXPHM
  "reference frequency": 20

likelihood:
  "minimum frequency": {H1: 20, L1: 20}
  "maximum frequency": {H1: 896, L1: 896}

priors:
  M_c: {type: uniform, minimum: 10.0, maximum: 80.0}
  q: {type: uniform, minimum: 0.125, maximum: 1.0}
  iota: {type: sine}
  dec: {type: cosine}
```

and an analysis blueprint attaching a Jim production to it (only settings
specific to *this* analysis — pipeline choice, scheduler resources, any
`jim:` overrides — belong here; everything above is inherited):

```yaml
# jim-analysis.yaml
kind: analysis
name: GW150914_jim
pipeline: jim
event: GW150914_095045

scheduler:
  cpus: 2
  gpus: 1
  "request memory": 8192MB
  "accounting group": ligo.dev.o4.cbc.pe.jim
```

```bash
asimov apply -f event.yaml
asimov apply -f jim-analysis.yaml
asimov manage build       # render this analysis's TOML config (ledger input -> config.toml, not the other way round)
asimov manage submit      # submit the jim-run job
asimov monitor            # poll for completion
```

### The `jim:` metadata block

Jim's own configuration (data source, waveform, sampler, output) doesn't map
cleanly onto a single generic Asimov vocabulary, since Jim supports several
data-loading modes (`gwosc`, `injection`, `file`) and five sampler backends
each with their own settings. Common fields (`interferometers`,
`"event time"`, `waveform.approximant`, `likelihood."minimum/maximum
frequency"`) are picked up automatically from whichever blueprint set them
(same as other pipelines), but anything Jim-specific is set under a `jim:`
block *in the ledger blueprint* (an `analysis` blueprint for a single
production, or an `event`/`configuration` blueprint if you want it shared
across several). Its sub-sections mirror
[Jim's own TOML config](https://gw-jax-team.github.io/Jim/stable/quickstart/)
almost exactly, and the plugin renders that ledger input into Jim's actual
`config.toml` file when you run `asimov manage build`:

```yaml
# added to an analysis (or event/configuration) blueprint
jim:
  seed: 0
  verbose: false

  data:
    type: gwosc               # gwosc (default) | injection | file
    # injection: sampling_frequency, duration, zero_noise, injection_parameters
    # file: duration, strain_files: {H1: ..., L1: ...}, psd_files: {...}, strain_channels: {...}, psd_is_asd: {...}

  waveform:
    approximant: IMRPhenomXPHM
    f_ref: 20.0

  likelihood:
    f_min: 20.0                # overrides the "likelihood: minimum frequency" derivation
    f_max: 896.0

  sampling:
    time_frame: detector        # geocentric | detector -- passed straight through to Jim's [sampling] section
    sky_frame: equatorial

  sampler:
    type: flowmc                # flowmc | blackjax-ns-aw | blackjax-nss | blackjax-swig | blackjax-smc
    n_chains: 1000
    n_local_steps: 100
    n_global_steps: 1000
    # any other field for the chosen sampler's config is passed straight through

  output:
    n_samples: 5000
    save_corner: true
    corner_parameters: [M_c, q]
    # any other Jim [output] field is passed straight through too

  prior:
    # Jim-native prior spec, used as-is instead of the priors: -> jim
    # conversion below. Only needed if you want a prior type the
    # conversion below doesn't cover (rayleigh, uniform_sphere).
    M_c: {type: uniform, min: 10.0, max: 80.0}
```

`checkpoint_dir`/`checkpoint_interval` and `output.dir`/`output.overwrite`
are always set by the plugin itself (to `<rundir>/checkpoint` and
`<rundir>/output` respectively, kept separate so Jim's own checkpoint
writes don't trip its refusal to overwrite an existing output directory)
and can't be overridden.

### Priors

A standard Asimov `priors:` block is converted into Jim's `[prior]` table
for the five prior types with an unambiguous match: `uniform`, `gaussian`,
`sine`, `cosine` and `power_law` (Jim's `rayleigh` and `uniform_sphere`
priors have no generic equivalent). **Parameter names are passed through
unchanged** -- write them using Jim's own names (`M_c`, `q`, `s1_z`,
`iota`, `d_L`, `t_c`, ...; see `jim-run --init` for the full reference
parametrization). This plugin deliberately does not attempt to translate
parameter names or spin/distance conventions from another pipeline's
convention (e.g. Bilby's `chirp_mass`/`mass_ratio`), since guessing at that
mapping risks silently producing a different physical prior than the one
written in the ledger.

An explicit `jim: prior:` block, if given, always takes precedence over the
`priors:` conversion.

### GPUs

Jim is JAX-based and normally wants a GPU. Set `scheduler: {gpus: N}` (as
in the example above) to request one -- `request_gpus` under HTCondor,
`--gres=gpu:N` under Slurm.

### Checkpointing and resumption

`jim-run` checkpoints its own progress to `checkpoint.pkl` and resumes
automatically whenever that file is present, with no special flag needed.
If a job is evicted or fails, `resurrect()` simply resubmits the same
command (up to 5 attempts), which picks up from the last checkpoint.

## Post-processing

Like the sibling [asimov-pycbc](https://github.com/etive-io/asimov-pycbc)
plugin, this plugin does not submit any post-processing job itself.
Express post-processing (e.g. PESummary) as a separate production with a
`needs:` dependency on the Jim production; Asimov's own dependency
resolution builds and submits it once the Jim production finishes, and it
can pick up the samples via `collect_assets()["samples"]`
(`<rundir>/output/samples.npz`).

## Development

```bash
pip install -e ".[test]"
pytest
```

See [asimov-plugin-template](https://github.com/etive-io/asimov-plugin-template)
for background on Asimov's plugin architecture, and the sibling
[asimov-pycbc](https://github.com/etive-io/asimov-pycbc) and
[asimov-bayeswave](https://github.com/etive-io/asimov-bayeswave) plugins
for other single-job (non-DAG) pipeline integrations that follow the same
pattern.

## License

MIT License - see [LICENSE](LICENSE) for details.
