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

### Settings: shared vocabulary first, `jim:` only for what's genuinely jim-specific

Most of Jim's configuration maps directly onto the same generic Asimov
vocabulary every other pipeline reads — there's no `jim:` override layer
for any of these:

- `interferometers`, `"event time"`, `data."data files"`, `data.channels`,
  `data."segment length"` → `[data]` (see [Data](#data) below — Jim always
  reads already-resolved frame files this way, never fetches its own)
- `waveform.approximant`, `waveform."reference frequency"` → `[waveform]`
- `likelihood."minimum/maximum frequency"` (lowest/highest across
  detectors) → `[likelihood] f_min`/`f_max`
- `priors:` → `[prior]`, via the standard `PriorInterface.convert()`
  mechanism (see [Priors](#priors) below)
- `sampler: {sampler: <name>, "sampler kwargs": {...}}` (the same shape
  `bilby`/`pycbc` use) → `[sampler] type` + whatever kwargs Jim's chosen
  backend takes, e.g.:

  ```yaml
  sampler:
    sampler: flowmc              # flowmc | blackjax-ns-aw | blackjax-nss | blackjax-swig | blackjax-smc
    "sampler kwargs":
      n_chains: 1000
      n_local_steps: 100
      n_global_steps: 1000
      # any other field for the chosen backend's config is passed straight through
  ```

A small `jim:` block, set *in the ledger blueprint* (an `analysis`
blueprint for a single production, or an `event`/`configuration` blueprint
if shared), covers the handful of things that have no generic Asimov
equivalent at all — PSD file paths (see [Data](#data)), Jim's `[sampling]`
section, output options, and an escape hatch for the two prior types the
generic `priors:` mechanism can't express:

```yaml
# added to an analysis (or event/configuration) blueprint
jim:
  seed: 0
  verbose: false

  data:
    psd_files: {H1: ..., L1: ...}   # no generic equivalent -- see Data below
    psd_is_asd: {H1: false, L1: false}

  sampling:
    time_frame: detector        # geocentric | detector -- passed straight through to Jim's [sampling] section
    sky_frame: equatorial

  output:
    n_samples: 5000
    save_corner: true
    corner_parameters: [M_c, q]
    # any other Jim [output] field is passed straight through too

  prior:
    # Jim-native prior spec, used as-is instead of the priors: -> jim
    # conversion below. Only needed for a prior type the conversion
    # doesn't cover (rayleigh, uniform_sphere).
    M_c: {type: uniform, min: 10.0, max: 80.0}
```

`checkpoint_dir`/`checkpoint_interval` and `output.dir`/`output.overwrite`
are always set by the plugin itself (to `<rundir>/checkpoint` and
`<rundir>/output` respectively, kept separate so Jim's own checkpoint
writes don't trip its refusal to overwrite an existing output directory)
and can't be overridden.

### Data

`asimov-jim` always uses Jim's `file` data mode and never Jim's own
built-in `gwosc`/`injection` fetch modes — this production must already
have real frame files resolved before it builds, exactly like every other
pipeline, so that the data used is consistent across pipelines in the same
project rather than each one fetching independently. Strain files come
from the generic `data:` blueprint vocabulary (the
[adding-a-pipeline tutorial](https://asimov.readthedocs.io/en/latest/tutorials/adding-a-pipeline.html)
and [GWOSC cookbook](https://asimov.readthedocs.io/en/latest/ligo-cookbook/working-with-gwosc.html)
show the canonical pattern), typically populated by a `pipeline: gwdata`
production wired up via `needs:`:

```yaml
kind: analysis
name: get-data
pipeline: gwdata
download:
  - frames
---
kind: analysis
name: GW150914_jim
pipeline: jim
event: GW150914_095045
needs:
  - get-data
```

```yaml
# resolved onto the event by the gwdata production above
data:
  "data files": {H1: /path/to/H1.gwf, L1: /path/to/L1.gwf}
  channels: {H1: H1:GWOSC-16KHZ_R1_STRAIN, L1: L1:GWOSC-16KHZ_R1_STRAIN}
  "segment length": 4
```

→ Jim's `strain_files`/`strain_channels`/`duration`. There's no generic
Asimov vocabulary for PSD files (pipelines that need them typically
estimate their own, e.g. via a separate BayesWave PSD-generation
production), so those stay under `jim: data: psd_files:`/`psd_is_asd:`
(shown above) until a shared convention exists.

### Priors

Like every other pipeline, Jim reads priors from the ledger's generic
`priors:` block, via a `JimPriorInterface.convert()` that translates each
entry's `type` into Jim's own `[prior]` table. It covers the five prior
types with an unambiguous field-for-field match: `uniform`, `gaussian`,
`sine`, `cosine` and `power_law`. Two of Jim's own prior types have no
equivalent in the generic `PriorSpecification` (`rayleigh`, `uniform_sphere`);
for those, set the parameter directly under `jim: prior:` (see above),
which takes precedence over the `priors:` conversion when given.

**Parameter names are passed through unchanged** -- write them using Jim's
own names (`M_c`, `q`, `s1_z`, `iota`, `d_L`, `t_c`, ...; see `jim-run
--init` for the full reference parametrization). This plugin deliberately
does not attempt to translate parameter names or spin/distance conventions
from another pipeline's convention (e.g. Bilby's `chirp_mass`/`mass_ratio`),
since guessing at that mapping risks silently producing a different
physical prior than the one written in the ledger.

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
