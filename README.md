# Tree Cracker[^1]

This is a fork of [Andrew (Gaider10)'s TreeCracker](https://github.com/Gaider10/TreeCracker). Compile the CUDA solver once, then pass tree observations in a text file for each search. Observation tables are constructed at startup using the existing tree model and chunk ordering. Search settings remain in [Settings](./Settings%20(MODIFY%20THIS).cuh).

## Purpose
Given details about a set of Minecraft trees (such as their coordinates, types, and attributes), this code is designed to return a list of <!-- worldseeds --> structure seeds[^2] that could <ins>potentially</ins> generate those exact trees.

## Prerequisites and Limitations
At the time of writing, this program only officially supports
- Java Edition.
- Versions <!-- 1.6.4, 1.8.9, 1.12.2, --> 1.14.4, 1.16.1, 1.16.4, or 1.17.1. (Informally, if a specific version isn't supported, setting the relevant data to the next chronological supported version&mdash;e.g. marking 1.16.2 as 1.16.4&mdash;will sometimes still yield results.)
- Oak (normal or fancy) <!--, Spruce, Pine, --> or Birch trees.
- Forest <!--, Birch Forest, or Taiga --> biomes.

The program also uses CUDA, which requires one's device to have an NVIDIA CUDA-capable GPU installed. NVIDIA's CUDA also [does not support MacOS versions OS X 10.14 or beyond](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts). If either of those requirements disqualify your computer, you can instead use a free notebook GPU, subject to availability and usage limits. See [the current free-GPU options and benchmark instructions](#free-gpu-access-after-colab-quota).

If using Windows, you will also need some form of C++ compiler installed; however, there are a myriad of environments that provide one ([Microsoft Visual C++](https://learn.microsoft.com/en-us/cpp/build/reference/compiler-options), though that in turn requires [Visual Studio](https://visualstudio.microsoft.com); [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl); [Minimialist GNU for Windows-w64](https://www.mingw-w64.org); and others).

## Installation, Setup, and Usage
1. Download the repository, either as a ZIP file from GitHub or by cloning it through Git.
2. Write one observation per line: `version Forest type x z height_min height_max leaf_states`. Types are `Oak`, `Fancy_Oak`, `Birch`, or `Unknown`. Heights are inclusive ranges; `0 0` means unknown. The twelve leaf characters are `0` (absent), `1` (placed), or `?` (unknown), ordered lowest/middle/highest layer, each Northwest/Southwest/Northeast/Southeast. Fancy Oak and Unknown require unknown leaf states. Blank lines and `#` comments are accepted. See [the fresh four-tree input](Test%20Data/fresh-16-1-20261009.txt). No seed is part of the input. Change [Settings](./Settings%20(MODIFY%20THIS).cuh) only when changing search configuration.

<!-- TODO: Rework warnings to apply these (i.e. warn about low first-tree and first-treechunk bits instead of total bits) -->
When creating your input data, keep in mind that <ins>the comprehensiveness of the data</ins> (specifically the number of "bits" of information the highest-information tree and treechunk <!-- TODO: Explain treechunks? --> reveal) <ins>matters far more than factors like the number of trees.</ins> For example, I and Chaos4669 once tried to crack the same worldseed using this tool:
 - my input data had eleven trees holding 123.87 combined bits of information, but my highest-information treechunk among that revealed only 54.71 bits and my highest-information tree only 13.06 bits&mdash;which caused the program to return dozens of possibilities and require multiple days before it could finish.
 - Chaos' input data ([`TEST_DATA_16_1_2`](./Test%20Data/1.16.1.cuh)), meanwhile, had half the number of trees and total bits of information, but his highest-information treechunk held 68.93 bits of information and his highest-information tree revealed 19.91 bits of information. With those, the program finished within five minutes and returned a mere four possibilities.

(The program will print the number of bits each piece of input data reveals when the program first begins running. If a multi-day runtime cannot be avoided, the program also comes with the ability to divide one's runs into "partial runs" so that one's run can be resumed midway-through at a later time.)

3. Go back and double-check your input data. There is an 80% chance you inputted something incorrectly the first time, and any mistakes will prevent the program from deriving the correct worldseeds.
4. Once you're *completely certain* your input data is correct&mdash;if you wish to run the program on Google Colab:
    1. Visit [the website](https://colab.research.google.com), sign in with a Google account, and create a new notebook.
    2. Upload or clone the program's files, retaining their directory structure, and upload your observations file.
    3. Under the Runtime tab, select "Change runtime type" and select T4 GPU as the hardware accelerator.
5. Whether on Google Colab or your own computer, open a terminal and verify [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html), the CUDA compiler, is installed:
```bash
(Linux/Windows/MacOS)  nvcc --version
(Google Colab)        !nvcc --version
```
If the output is an error and not the compiler's information, you will need to install the CUDA Toolkit which contains `nvcc`. (The following are installation guides for [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux), [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows), and [MacOS X 10.13 or earlier](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-mac-os-x/).)

6. Navigate in the terminal to the folder where the program's files are contained:
```bash
(Linux/Windows/MacOS)  cd "[Path to the folder]"
(Google Colab)        !cd "[Path to the folder]"
```
Then use `nvcc` to compile the program:
```bash
(Linux/T4)      nvcc main.cu -o main -O3 -arch=sm_75 -Xcompiler=-mcmodel=large -Xlinker=--no-relax --cudart=shared
(Windows)       nvcc main.cu -o "main.exe" -O3
(MacOS)         nvcc main.cu -o "main.app" -O3
(Google Colab) !nvcc main.cu -o main -O3 -arch=sm_75 -Xcompiler=-mcmodel=large -Xlinker=--no-relax --cudart=shared
```
Compilation is a one-time cost for each search configuration and GPU target; changing observations does not require rebuilding.<br />
The compiler may print warnings akin to `Stack size for entry function '_Z11biomeFilterv' cannot be statically determined`: this is normal. (All this means is that the compiler couldn't determine the exact number of iterations certain recursive functions will undergo.)

7. Run the compiled program:
```bash
(Linux)         LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./main observations.txt results.txt
(Windows)       .\main.exe observations.txt results.txt
(MacOS)         ./main.app observations.txt results.txt
(Google Colab) !LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ./main observations.txt results.txt
```
As mentioned in step 2, the program's runtime can vary wildly based on one's input data and its comprehensiveness. Nevertheless, if all goes well, a file should ultimately be created (or a list should be printed to the screen, depending on your settings) containing your possible <!-- worldseeds --> structure seeds.

8. At some point, this program will also automatically filter structure seeds into potential worldseeds. This hasn't been implemented yet, though, so in the meantime one must perform this filtering manually.
    1. Download and open [Cubiomes Viewer](https://github.com/Cubitect/cubiomes-viewer/releases).
    2. Under the Edit tab in the upper top-left, click "Advanced World Settings" and make sure "Enable experimentally supported versions" is enabled.
    3. Close the World Settings menu and set the "MC" input box in the top-left corner to your world's version (or the supported version closest to it).
    4. Under the Seed Generator heading, Click "Seed list", then use the button across from the "Load 48-bit seed list" option to select whichever file contains this program's outputted structure seeds.
    5. For each population chunk in your input data (these will have been displayed when <ins>this</ins> program first began running):
        - Under the Conditions heading, click "Add".
        - Select "Biomes" for the condition's category and "Overworld at scale" as the condition's type.
        - Select Custom for the location and enter the population chunk's coordinate range.
        - Select "1:1 ..." for the Scale/Generation Layer, then exclude all biomes except the population chunk's biome.
    6. When finished adding all conditions, click "Start search" at the bottom of the window. The program will then start outputting worldseeds that have biomes matching your input data.

WARNING: When checking the outputted worldseeds, some generated trees may not match your input data. (Tree generation depends on the order that chunks are loaded, so if the chunks are loaded in a different order than your input data's source, a different pattern of trees will form.) However, in most cases at least a few trees will match your input data; if *every* tree is different, that is an indication your original input data (or this tool) are likely wrong.

## Java 1.16.1 CUDA benchmark (October 2026)

This fork optimizes the existing Java-LCG and valid/invalid tree-attempt search. Production inference still searches the entire 2^44-state absolute-coordinate domain. The tree RNG model and acceptance predicates are unchanged. The integrated changes enumerate exact primary Z intervals, share coordinate work in the neighboring-state scans, and share/prune valid/invalid attempt prefixes in the treechunk filter.

Two existing correctness issues were repaired before performance comparisons: host access to device-only result buffers could discard candidates, and 32-bit treechunk indexing could omit states in large launches. Comparisons use these repairs consistently and reject every overflowing run. The old Linux port returned one candidate on `TEST_DATA_16_1_2`; corrected host access returns the documented four candidates, including `157527116063087`.

All measurements used the same allocated Tesla T4: UUID `GPU-2891ecda-a53e-c564-6771-d543407075ff`, driver 580.82.07, CUDA 13.0.88. Compilation is excluded from solve times. Complete measurements, pinned source commits, flags, input hashes, test windows and candidate sets are in [the five-seed report](tests/results/five-seed-t4.json), [the combination report](tests/results/combined-scan-t4.json), and [the exhaustive reference report](tests/results/reference-t4.json).

The five fixtures contain two public examples and three instrumented development-world Forest chunks. Each test input has absolute X/Z and tree type; heights and leaf corners are unknown. Development chunk biomes were verified from their development saves. No blind records were opened. These tests validate the standalone cracker with supplied tree observations; they do not measure video extraction or blind evaluation.

The following are median **validation-only regression** pipeline times in seconds, three repeats per fixture, over identical windows of 2^21 states with capacity 2^22. Known seeds locate positive windows only in the test executable, so every run must retain a real expected seed. Production inference never reads those validation seeds. These timings include stage-by-stage test checks and must not be interpreted as exhaustive seed recovery times.

| Variant | Public 4 trees | Public 5 trees | Dev 9 trees | Dev 8 trees | Dev 10 trees |
|---|---:|---:|---:|---:|---:|
| Corrected baseline | 10.908249 | 1.158632 | 0.083872 | 0.410807 | 0.059132 |
| Shared scan | 10.910895 | 1.159273 | 0.063400 | 0.418232 | 0.049571 |
| Primary intervals | 10.908272 | 1.159967 | 0.079580 | 0.412385 | 0.057368 |
| 128-thread blocks / sort | 10.909658 | 1.162082 | 0.080481 | 0.415300 | 0.055784 |
| Primary + prefix search | 0.217338 | 0.068668 | 0.053583 | 0.067597 | 0.048405 |
| Primary + prefix search + scan | 0.217542 | 0.065370 | 0.043956 | 0.054954 | 0.057434 |

All 90 GPU regression runs passed with identical sorted candidate sets and all five expected seeds retained. The per-fixture candidate counts were 27, 4, 1, 2 and 3. Sixteen CPU tests cover interval arithmetic, Java-LCG scan parity, 64-bit dispatch and prefix-search equivalence.

The complete, documented `TEST_DATA_16_1_2` example retains its trunk-height and leaf-corner observations. On the same T4, corrected baseline took **232.033 s**, primary intervals **105.001 s**, primary plus prefix search **72.310 s**, and adding the shared scan **65.881 s median** over three full runs (65.874–70.990 s), a **3.52×** speedup over the corrected baseline. Every exhaustive run returned exactly `[89593286004526, 115468527086863, 157527116063087, 272740909957168]`. These are 48-bit structure seeds; no unique 64-bit world seed is claimed.

An initial four-tree positions/types-only diagnostic took 87.235 s for its first 2^24 primary states before the 32-bit treechunk indexing defect was found. Its candidate set was incomplete and is excluded from correctness comparisons. An exhaustive run would take years if that diagnostic rate persisted. Five exhaustive positions/types-only recoveries were not performed; the five positive regression tests and the rich full-domain example are reported separately.

Reproduce the fixed five-seed comparisons from this branch on one CUDA GPU:

```bash
python3 tests/benchmark.py --positive-window --workers 2097152 --capacity 4194304 --repeats 3 \
  --variant baseline=87d3675 --variant scan=b231ecf --variant primary=cf41a29 \
  --variant host=2d338c1 --variant primary-dfs=9ddaf10 --output benchmark-results/five-seed
python3 tests/benchmark.py --positive-window --workers 2097152 --capacity 4194304 --repeats 3 \
  --variant combined-scan=de9371e --reference-report benchmark-results/five-seed/report.json \
  --output benchmark-results/combined-scan
python3 -m unittest discover -s tests -v
```

The output path must be new. Omit `--positive-window` for the exhaustive positions/types-only search, which requires far more GPU time. `--profile-batches` explicitly selects an incomplete first-window diagnostic. The reports label these modes and refuse comparisons after the GPU allocation or benchmark configuration changes.

For the documented full example, remove the placeholder `INPUT_DATA` array from the Settings file and define `INPUT_DATA` as `TEST_DATA_16_1_2`. Keep the existing default domain, batch size and sixteen partial runs. On the tested Linux/Colab CUDA 13 environment:

```bash
nvcc main.cu -o main -O3 -arch=sm_75 -Xcompiler=-mcmodel=large -Xlinker=--no-relax --cudart=shared
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./main 'Test Data/documented-16-1.txt'
```

## Fresh vanilla 1.16.1 world recovery

A new, unrestricted random vanilla Java 1.16.1 world was generated on October 9, 2026 with the existing recorder. The standalone optimized solver received tree observations only, with no known seed or restricted validation window. Its complete 2^44-state absolute-coordinate search took **54.036 s** on the same allocated T4 used above and returned exactly one structure seed, **128110318218222**. The separate compilation took **60.601 s**. The candidate equals the lower 48 bits of the new world's validation-only world seed, `-2106430615384331282`. This is one measured full run.

The input used **four birch trees across two verified Forest chunks**, with exact absolute X/Z, exact trunk heights, and all twelve corner-leaf states per tree (**48 corner states**):

| Tree | X | Z | Trunk height | Population chunk |
|---|---:|---:|---:|---|
| Birch | 7 | 26 | 5 | (0, 1) |
| Birch | 4 | 19 | 6 | (0, 1) |
| Birch | 15 | 24 | 7 | (0, 1) |
| Birch | 9 | 35 | 7 | (0, 2) |

The first three trees produced six structure-seed candidates; the fourth tree in the neighboring chunk reduced these to one. Four trees were sufficient for this input; the absolute minimum was not tested. These observations were collected directly from the generated world, so this measurement excludes video extraction and data collection time. No blind records were read.

The exact observations are saved in [the runtime input](Test%20Data/fresh-16-1-20261009.txt), with hardware, compile flags, input hash, seed verification and per-tree details in [the measurement](tests/results/fresh-world-t4.json) and all sixteen partitions in [the run log](tests/results/fresh-world-t4.log). The historical compiled-input measurement used revision `12b52097eb136dd32074b1435288a51df688e09a` and [this input array](Test%20Data/fresh-16-1-20261009.cuh). The current executable takes the text input directly.

## Runtime observations: total time to 48-bit output

The reusable runtime-input binary was compared with compiled observations on the **same T4 UUID above**, using identical CUDA flags, inputs and complete 2^44-state searches. Each compiled-input trial included observation preparation, a fresh compilation, process/CUDA startup, search and output. Runtime-input trials reused one binary and included writing the observation file, parsing, chunk construction, startup, search and output. Source retrieval, GPU provisioning and collecting tree observations were outside both timers. Orders alternated across three repeats per input.

| Input | Compiled: complete output | Runtime: complete output | Total speedup |
|---|---:|---:|---:|
| Fresh four-tree world | 121.494 s | 60.892 s | 1.995× |
| Documented five-tree example | 131.526 s | 72.435 s | 1.816× |

For the fresh world, the correct **128110318218222** candidate was first emitted after **44.099 s** with runtime input, versus **105.354 s** including per-input compilation. Completion of the exhaustive search confirmed it was unique. The documented example preserved its exact four-candidate set; its first candidate alone does not establish the correct seed.

The reusable build took **79.303 s once**. Including this build, a first fresh-world runtime-input search totals **140.195 s**; later fresh-world inputs take the measured **60.892 s** through complete output. Search-process medians rose from 56.840 to 60.892 s for the fresh case (about 7.1%), and from 70.287 to 72.435 s for the documented case (about 3.1%). Eliminating input-specific compilation outweighs this cost once the binary is built. These observations still include heights and leaf corners; this comparison does not measure a positions/types-only exhaustive search.

All **12 exhaustive comparison runs** preserved exact output, with no overflow. The runtime loader also passed **15/15 bounded GPU regressions** across the same five fixtures, matching earlier candidate sets and exact known-positive windows with one reusable binary. Those bounded regressions validate correctness; they are not full seed-recovery timings. The local parser and arithmetic suite passed 20 tests. No blind data was read, and no seed or restricted window was supplied to either exhaustive solver.

Pinned revisions, trial times, input/binary hashes and timing boundaries are in [the total-time report](tests/results/runtime-input-t4.json). The [five-seed report](tests/results/runtime-five-seed-t4.json), [fresh runtime log](tests/results/runtime-input-t4.log) and [visible T4 proof](tests/results/runtime-input-t4-proof.jpg) provide validation evidence. To reproduce on a T4 with the repository available:

```bash
python3 tests/benchmark_runtime.py --baseline a69f236 --repeats 3 --output benchmark-results/runtime-input
python3 tests/benchmark.py --positive-window --workers 2097152 --capacity 4194304 --repeats 3 \
  --variant runtime=b2568e0 --reference-report tests/results/combined-scan-t4.json \
  --output benchmark-results/runtime-five-seed
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Free GPU access after Colab quota

Checked October 9, 2026. **Kaggle Notebooks with the T4 x2 accelerator is the recommended next free runtime for this solver.** It supplies the same GPU model as our Colab measurements, but a different physical allocation: baseline and optimized binaries must both be measured again in the same new session.

| Service | Current free access | Suitability for this run |
|---|---|---|
| [Kaggle Notebooks](https://www.kaggle.com/code) | [Weekly GPU allowance of 30 hours, sometimes higher](https://www.kaggle.com/docs/efficient-gpu-usage); [GPU sessions up to 12 hours](https://www.kaggle.com/docs/notebooks). Check the account's actual remaining allowance in notebook settings. | Best next option. [T4 x2 remains available; P100 was retired September 15, 2026](https://www.kaggle.com/product-announcements/735239). Each T4 has 16 GB of memory. |
| [Google Colab](https://research.google.com/colaboratory/faq.html) | Free GPU access depends on demand and previous usage; Google does not publish fixed usage limits. | Our replacement GPU request was denied for usage limits on October 9. The dialog provided no recovery countdown. **No reliable reset time or guaranteed 24-hour wait can be given.** |
| [Amazon SageMaker Studio Lab](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html) | Existing customers can use GPU sessions up to four hours, capped at four hours per 24 hours, subject to availability. | AWS now says Studio Lab is closed to new customers, so it is useful only if an account already exists. |

### Run the pending comparison on Kaggle

1. Sign in to Kaggle, create a private Python notebook, and complete any account verification requested for GPU access. Select **Settings → Accelerator → GPU T4 x2**, and enable Internet so the notebook can clone the fork. An allocated GPU and a working `nvcc` compiler are required; setup on Kaggle has not yet been validated.
2. Run the following cell once in a fresh session. It uses the existing benchmark programs, pins both source revisions, and exposes only the first physical GPU to every CUDA process. The second T4 is unused by this single-GPU solver.

```python
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

subprocess.run(["nvcc", "--version"], check=True)
hardware = subprocess.check_output([
    "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
    "--format=csv,noheader",
], text=True).strip()
print(hardware, flush=True)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = hardware.splitlines()[0].split(",")[0].strip()

repo = Path("/kaggle/working/tree-cracker")
subprocess.run([
    "git", "clone", "https://github.com/YHDiamond/tree-cracker.git", str(repo),
], check=True)
# This experimental revision contains the runtime-vs-runtime benchmark.
subprocess.run(["git", "switch", "--detach", "344b6de"], cwd=repo, check=True)
subprocess.run([
    "python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py",
], cwd=repo, env=env, check=True)

results = Path("/kaggle/working") / (
    "tree-benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
)
variants = ["--variant", "baseline=8242664", "--variant", "optimized=344b6de"]
# Five seeds, three repeats per variant; bounded validation-only windows.
subprocess.run([
    "python3", "tests/benchmark.py", *variants,
    "--positive-window", "--workers", "2097152", "--capacity", "4194304",
    "--repeats", "3", "--output", str(results / "five-seed"),
], cwd=repo, env=env, check=True)
# Two rich observation sets, three alternating repeats per variant;
# complete searches, with no known seed supplied to either solver.
subprocess.run([
    "python3", "tests/benchmark_runtime.py", *variants,
    "--repeats", "3", "--expected-hardware", hardware,
    "--output", str(results / "full-search"),
], cwd=repo, env=env, check=True)
print("Reports and logs:", results, flush=True)
```

3. Download the reports and logs from the printed directory before ending the session. They are written under Kaggle's `/kaggle/working` output directory. Stop the GPU session when finished to conserve the allowance. Repeating the cell requires a new clone directory and new output paths; the benchmark programs intentionally refuse existing output directories.

The five-seed command runs **30 bounded regression trials**, compares exact sorted candidate sets and the same positive windows, and retains each fixture's expected seed. The optimized test executable also checks the new leaf, Birch-selection and rewind predicates against independent reference calculations on the GPU. These tests establish output parity in the tested windows; their times do not represent full recoveries.

The full-search command runs **12 exhaustive trials** across the fresh four-tree world and documented five-tree example. Its primary metric is **complete observations available → correct 48-bit seed printed**, including writing the input, parsing, process/CUDA startup, search and output. Each reusable binary is compiled once before these timers; build time is recorded separately. GPU allocation, downloading the source and collecting tree observations are outside the timers. Validation truth is checked after each exhaustive run and never restricts the search or stops it early. Complete-output time is also recorded; the documented example remains ambiguous with four candidates.

Do not use the old Colab report as `--reference-report` on Kaggle: that option requires the same physical GPU UUID and configuration. Compare both revisions within the new allocation. Both programs reject candidate differences, unsuccessful runs and overflow; the full-search program also rejects hardware changes.

### Optimization status when Colab stopped

The released implementation and its completed multi-seed evidence are committed on the fork's default branch, **`benchmark-main`**. The next optimizations are pushed separately on **`opt/runtime-speed`**, pinned above at `344b6de`, and await the remaining GPU validation. In the first paired fresh-world trial, the correct seed appeared after **33.398 s optimized versus 39.610 s baseline**, with identical unique output; complete searches took 46.632 s and 54.986 s respectively. This is one paired trial, not a validated median improvement. Colab ended the allocation before the multi-seed checks completed, and no Kaggle performance result is claimed yet.

## Acknowledgements
I would like to give very large Thank You's to
- [Andrew](https://github.com/Gaider10), for creating the [original version of the TreeCracker](https://github.com/Gaider10/TreeCracker) (alongside much of the test data) and a [population chunk reverser](https://github.com/Gaider10/PopulationCrr), and for answering a question about his tool.
- [Cortex](https://github.com/mcrcortex) and [TatertotGr8](https://github.com/tatertotgr8), for their tree crackers that predate even Andrew's ([Cortex's](https://github.com/MCRcortex/TreeCracker), [TatertotGr8's](https://github.com/TatertotGr8/Treecracker)).
- [Epic10l2](https://github.com/epic10l2), for his [comprehensive guide to Andrew's TreeCracker](https://docs.google.com/document/d/1csrcO2F4qQ2ahYgcicWmJtnfeU99q65p) that enabled me to learn how to use the program.
- [Edd](https://github.com/humhue), for helping considerably with Epic10l2's guide, and for answering a few questions about 1.12.2- population reversal.
- [Neil](https://github.com/hube12), for finding and listing most tree salts ([1.13](https://gist.github.com/hube12/574512a3c4df2be8ba6c08e7298caedd), [1.14](https://gist.github.com/hube12/394ddf11b3cdcc9504270777565446e4), [1.15](https://gist.github.com/hube12/821b66615a97a7130ef804603d68bec8), [1.16](https://gist.github.com/hube12/b65500cd234ce2a3983b62b3903c183d), [1.17](https://gist.github.com/hube12/5066fbcd8565648dd68113a9b065514b)).
- [Chaos4669](https://youtube.com/@Chaotic4669), for providing test data and recommendations on what could benefit from clarification.
- [Cubitect](https://github.com/cubitect), for his [Cubiomes library](https://github.com/Cubitect/cubiomes) that this program (will ultimately) use a port of to filter biomes, and his [Cubiomes Viewer](https://github.com/Cubitect/cubiomes-viewer) GUI tool I recommend as a substitute in the meantime.
- [Panda4994](https://github.com/panda4994), for [his algorithm]((https://github.com/Panda4994/panda4994.github.io/blob/48526d35d3d38750102b9f360dff45a4bdbc50bd/seedinfo/js/Random.js#L16)) to determine if a state is derivable from a nextLong call.

If you would like to contribute to this repository or report any bugs, please feel free to open an issue or a pull request.

This repository is offered under [my (NelS') general seedfinding license](https://github.com/Nel-S/seedfinding/blob/main/LICENSE). Please read and abide by that text if you have any wishes of referencing, distributing, selling, etc. this repository or its code.[^3]

[^1]: ...or more accurately "Tree Brute-forcer", but the previous term is so ingrained in seedcracking culture that I can't exactly change it now.
[^2]: If one converts a worldseed into a 64-bit binary integer, a structure seed corresponds to the worldseed's last 48 bits. Therefore each structure seed has 2<sup>16</sup> = 65536 worldseeds associated with it. ...eventually, biome filtering will be used to directly return worldseeds instead of structure seeds, but this has not been finished yet.
[^3]: While the license discusses this, I want to emphasize one aspect of it here: this repository relies upon numerous others' repositories (Gaider10's TreeCracker, Gaider10's Population Chunk Reverser, Cubitect's Cubiomes library, etc.), and thus my license solely applies to the changes I and any voluntary contributors made within this repository, not to their repositories or any code in this repository that is untouched from their repositories.
