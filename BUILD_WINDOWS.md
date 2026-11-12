# Building SageAttention2++ on Windows for ComfyUI

This guide explains how to build and install SageAttention 2.2.0 (SageAttention2++) from source on
Windows 10/11, into the Python environment of a ComfyUI install.

The upstream `setup.py` was written only for Linux and does not build on Windows. This repo
includes a small `setup.py` patch, described in [What was changed](#what-was-changed-and-why), that fixes
this. The CUDA/C++ kernel sources need no changes.

Tested on Windows 11 with Python 3.13, PyTorch 2.13.0+cu130, CUDA Toolkit 13.5, Visual Studio
Community 2026 (MSVC 14.51), and an RTX 5090 (sm_120).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **NVIDIA GPU** | Ampere or newer (RTX 30xx / A100 = sm_80/86, RTX 40xx = sm_89, H100 = sm_90, RTX 50xx = sm_120). |
| **ComfyUI with a CUDA build of PyTorch** | Note the PyTorch CUDA version (e.g. `cu128`, `cu130`). |
| **CUDA Toolkit** | Its **major** version must match PyTorch's CUDA major version (e.g. torch `cu130` → CUDA 13.x toolkit; torch `cu128` → CUDA 12.8+). RTX 50xx needs CUDA ≥ 12.8; RTX 40xx needs ≥ 12.4. Download from NVIDIA. |
| **Visual Studio (2022 or 2026)** with the **"Desktop development with C++"** workload | Community edition is fine. You don't need a Developer Command Prompt: setuptools finds MSVC automatically. The MSVC version must be one your CUDA Toolkit supports. |
| **Python headers/libs** | The ComfyUI Python must have `include\Python.h` and `libs\pythonXY.lib` available (see [Troubleshooting](#troubleshooting) for the *portable* ComfyUI). |
| **Git** | To clone this repo. |

## 2. Check your environment

In the commands below, `PY` is the ComfyUI Python. Typical locations:

- venv-based install: `<ComfyUI>\.venv\Scripts\python.exe`
- portable install: `<ComfyUI_windows_portable>\python_embeded\python.exe`

Git Bash:

```bash
PY=/c/path/to/ComfyUI/.venv/Scripts/python.exe     # <-- change this
"$PY" -c "import sys,torch;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda);print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
nvcc --version
```

Check that:
- `torch.version.cuda` and `nvcc --version` have the **same major version** (13 and 13, or 12 and 12).
- `nvcc` is found. If not, add `<CUDA>\bin` to `PATH`, or set `CUDA_HOME` to the toolkit folder
  (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`).

## 3. Get the source with the Windows fix

Clone this repository, which already contains the patched `setup.py`:

```bash
git clone <this-repo-url> SageAttention
cd SageAttention
```

If you start from the upstream repo (`https://github.com/thu-ml/SageAttention`) instead, apply the
changes listed in [What was changed](#what-was-changed-and-why) to `setup.py` first.

## 4. Install build helpers into the ComfyUI Python

```bash
"$PY" -m pip install ninja packaging wheel setuptools
```

`ninja` is optional but makes the build much faster (it compiles the ~10 CUDA files in parallel).

> If the ComfyUI venv was created by `uv` and has no `pip`, use
> `uv pip install --python "$PY" ninja packaging wheel setuptools` instead.

## 5. Build and install

Run this from the `SageAttention` folder. The venv's `Scripts` folder must be on `PATH` so PyTorch can
find `ninja`.

**Git Bash:**

```bash
PY=/c/path/to/ComfyUI/.venv/Scripts/python.exe     # <-- change this
export PATH="$(dirname "$PY"):$PATH"
export MAX_JOBS=8                                   # parallel compile jobs; lower it if you run out of RAM
"$PY" -m pip install --no-build-isolation -v .
```

**PowerShell:**

```powershell
$PY = "C:\path\to\ComfyUI\.venv\Scripts\python.exe"   # <-- change this
$env:PATH = "$(Split-Path $PY);$env:PATH"
$env:MAX_JOBS = "8"
& $PY -m pip install --no-build-isolation -v .
```

Notes:
- **Use `pip install --no-build-isolation .`, not `python setup.py install`.** `setup.py install` is
  deprecated. `--no-build-isolation` is required because the build has to use the PyTorch already
  installed in ComfyUI, and because `pyproject.toml` pins old `setuptools` versions that would
  otherwise be downloaded into an isolated environment.
- The build compiles only for the GPU(s) in the machine. To build for other GPUs (e.g. to build on one
  PC and copy to another), set `TORCH_CUDA_ARCH_LIST` before building, e.g.
  `export TORCH_CUDA_ARCH_LIST="8.9;12.0"` (Git Bash) or `$env:TORCH_CUDA_ARCH_LIST="8.9;12.0"` (PowerShell).
- The build takes 10–20 minutes without ninja and a few minutes with it. Thousands of `warning`
  lines are normal. Success ends with `Successfully installed sageattention-2.2.0`.
- A warning like `Error checking compiler version for cl ... FileNotFoundError` is harmless. It only
  means `cl.exe` isn't on `PATH`; setuptools still finds MSVC itself.

## 6. Verify

Run this from a folder **other than** the SageAttention source folder, so Python imports the installed
package rather than the source tree:

```bash
cd ~
"$PY" -c "
import torch, torch.nn.functional as F
from sageattention import sageattn
q,k,v = (torch.randn(2,24,4096,128,device='cuda',dtype=torch.float16) for _ in range(3))
ref = F.scaled_dot_product_attention(q,k,v)
out = sageattn(q,k,v,tensor_layout='HND')
print('cosine similarity vs SDPA:', F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item())
"
```

A cosine similarity of about **0.999** means it works. Values slightly below 1 are expected because
SageAttention uses quantized (INT8/FP8) attention.

## 7. Use it in ComfyUI

Start ComfyUI with the `--use-sage-attention` flag, e.g.:

```bash
"$PY" main.py --use-sage-attention
```

The startup log should say that sage attention is being used.

**Rebuild SageAttention** whenever you upgrade PyTorch, the CUDA Toolkit, or Python in that ComfyUI
install. The compiled modules are tied to those exact versions.

---

## What was changed and why

All changes are in `setup.py` and only apply when building on Windows (`sys.platform == "win32"`).
The Linux build is unchanged.

| # | Problem on Windows | Symptom | Fix |
|---|---|---|---|
| 1 | The C++ flags were GCC-only (`-g -O3 -fopenmp -lgomp -std=c++17`). MSVC ignores or misreads them, and they conflict with the C++20 standard PyTorch's headers require. | Hundreds of errors like `C2429: language feature 'nested-namespace-definition' requires compiler flag '/std:c++17'`, `'optional': is not a member of 'std'` | On Windows, use `CXX_FLAGS = ["/O2", "/Zc:preprocessor", "/Zc:__cplusplus", "-DENABLE_BF16"]`. Don't set `-std`; PyTorch adds `/std:c++20` / `-std=c++20` itself. |
| 2 | The CCCL headers in recent CUDA Toolkits (12.8+/13.x) refuse to compile with MSVC's legacy preprocessor. | `fatal error C1189: #error: MSVC/cl.exe with traditional preprocessor is used ... pass /Zc:preprocessor` | Add `-Xcompiler /Zc:preprocessor` and `-Xcompiler /Zc:__cplusplus` to `NVCC_FLAGS`, and remove nvcc's `-std=c++17` (it conflicts with PyTorch's `-std=c++20`). `-Xptxas=-v` (very noisy register-usage output) is also dropped. |
| 3 | `-D_GLIBCXX_USE_CXX11_ABI` is a GNU libstdc++ setting. | Not an error on its own, but meaningless with MSVC. | Only added on Linux. |
| 4 | The sm90 (H100) extension linked the CUDA driver with `-lcuda`, which is GCC linker syntax. | Link failure when building for sm_90. | On Windows use `libraries=["cuda"]` (links `cuda.lib` from the CUDA Toolkit). |
| 5 | `setup.py` builds several extensions in parallel threads. On Windows, PyTorch's `BuildExtension` temporarily patches a shared compiler object per extension, so parallel builds mix up flags between extensions (e.g. the wrong `-DTORCH_EXTENSION_NAME`). | Compile lines for one module containing another module's name. This can produce `.pyd` files that fail to import. | Force `parallel = 1` on Windows. Files *within* each extension are still compiled in parallel by ninja (`MAX_JOBS`). |

### The patch

<details>
<summary>Full diff against upstream <code>setup.py</code> (thu-ml/SageAttention, commit d1a57a5)</summary>

```diff
@@ -47,18 +47,39 @@ if not SKIP_CUDA_BUILD:
     # Supported NVIDIA GPU architectures.
     SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "10.0", "12.0", "12.1"}
 
+    IS_WINDOWS = sys.platform == "win32"
+
     # Compiler flags.
-    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
-    NVCC_FLAGS = [
-        "-O3",
-        "-std=c++17",
-        "-U__CUDA_NO_HALF_OPERATORS__",
-        "-U__CUDA_NO_HALF_CONVERSIONS__",
-        "--use_fast_math",
-        "--threads=8",
-        "-Xptxas=-v",
-        "-diag-suppress=174",
-    ]
+    if IS_WINDOWS:
+        # MSVC does not understand the GCC-style flags used on Linux (-g, -O3,
+        # -fopenmp, -lgomp, -std=...). PyTorch's BuildExtension already adds the
+        # language standard it requires (/std:c++20 for cl, -std=c++20 for nvcc),
+        # so no -std flag is set here. /Zc:preprocessor is required by the CCCL
+        # headers shipped with recent CUDA toolkits, and /Zc:__cplusplus makes
+        # MSVC report the real C++ standard in the __cplusplus macro.
+        CXX_FLAGS = ["/O2", "/Zc:preprocessor", "/Zc:__cplusplus", "-DENABLE_BF16"]
+        NVCC_FLAGS = [
+            "-O3",
+            "-Xcompiler", "/Zc:preprocessor",
+            "-Xcompiler", "/Zc:__cplusplus",
+            "-U__CUDA_NO_HALF_OPERATORS__",
+            "-U__CUDA_NO_HALF_CONVERSIONS__",
+            "--use_fast_math",
+            "--threads=8",
+            "-diag-suppress=174",
+        ]
+    else:
+        CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
+        NVCC_FLAGS = [
+            "-O3",
+            "-std=c++17",
+            "-U__CUDA_NO_HALF_OPERATORS__",
+            "-U__CUDA_NO_HALF_CONVERSIONS__",
+            "--use_fast_math",
+            "--threads=8",
+            "-Xptxas=-v",
+            "-diag-suppress=174",
+        ]
 
     # Append flags from env if provided
     cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
@@ -68,9 +89,11 @@ if not SKIP_CUDA_BUILD:
     if nvcc_append:
         NVCC_FLAGS += nvcc_append.split()
 
-    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
-    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
-    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
+    if not IS_WINDOWS:
+        # libstdc++ ABI flag; meaningless with MSVC.
+        ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
+        CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
+        NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
 
     if CUDA_HOME is None:
         raise RuntimeError(
@@ -210,7 +233,8 @@ if not SKIP_CUDA_BUILD:
                     "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
                 ],
                 extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
-                extra_link_args=['-lcuda'],
+                # Link against the CUDA driver API (cuda.lib on Windows, libcuda.so on Linux).
+                **({"libraries": ["cuda"]} if IS_WINDOWS else {"extra_link_args": ['-lcuda']}),
             )
         )
 
@@ -237,6 +261,14 @@ if not SKIP_CUDA_BUILD:
     # Defaults if not provided
     if parallel is None:
         parallel = 4
+    if IS_WINDOWS:
+        # On Windows, PyTorch's BuildExtension temporarily monkey-patches the shared
+        # compiler object per extension (to inject flags such as
+        # -DTORCH_EXTENSION_NAME). Building extensions in parallel threads races on
+        # that patch and mixes up flags between extensions, so build them one at a
+        # time. Sources within each extension are still compiled in parallel by
+        # ninja (MAX_JOBS) when ninja is installed.
+        parallel = 1
     # Ensure MAX_JOBS for underlying tooling if not explicitly set
     os.environ.setdefault('MAX_JOBS', '32')
```

</details>

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot find CUDA_HOME` | Install the CUDA Toolkit, and put its `bin` on `PATH` or set `CUDA_HOME`. |
| `The detected CUDA version (X) mismatches the version that was used to compile PyTorch (Y)` | Toolkit and PyTorch CUDA **major** versions differ. Install a matching toolkit (or a matching PyTorch build). |
| `CUDA 12.8 or higher is required for compute capability 12.0` | RTX 50xx needs a newer CUDA Toolkit. |
| `No target compute capabilities` | No GPU detected (e.g. building on a machine without one). Set `TORCH_CUDA_ARCH_LIST`. |
| `Microsoft Visual C++ 14.0 or greater is required` / `cl.exe` not found by setuptools | Install Visual Studio with the "Desktop development with C++" workload. |
| `unsupported Microsoft Visual Studio version!` from nvcc | Your MSVC is newer than your CUDA Toolkit supports. Update the CUDA Toolkit, or install an older MSVC toolset through the Visual Studio Installer. As a last resort, add `-allow-unsupported-compiler` via `export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"`. |
| `Cannot open include file: 'Python.h'` or `LINK : fatal error LNK1104: cannot open file 'python3XX.lib'` | Common with the **portable** ComfyUI (`python_embeded` has no headers/libs). Install the *same* Python version from python.org and copy its `include` and `libs` folders into `python_embeded`. |
| Build is very slow / compiles one file at a time | `ninja` not installed, or the Python `Scripts` folder isn't on `PATH`. Check with `"$PY" -c "from torch.utils.cpp_extension import is_ninja_available as n; print(n())"`, which should print `True`. |
| Compiler crashes / out of memory | Lower `MAX_JOBS` (e.g. `4`). |
| `ImportError: DLL load failed` when importing `sageattention` | Import `torch` first (ComfyUI does this automatically). Make sure the build used the same Python/PyTorch as ComfyUI, and rebuild after PyTorch upgrades. |
| `Unsupported CUDA architecture` at runtime | GPU older than Ampere (sm_80), which SageAttention2 doesn't support. |
| Verification imports the wrong package | You ran Python inside the source folder. `cd` elsewhere and retry. |
