#requires -Version 5.1
<#
.SYNOPSIS
Build and verify this checkout's custom Mitsuba on Windows.
.DESCRIPTION
Run -CheckOnly first to inspect prerequisites without installing or building.
Relative directory arguments resolve against the repository's parent workspace.
PythonExecutable is an optional Python 3.12 interpreter used to create a new venv.
Existing environments and build directories are never deleted or replaced.
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [ValidateRange(0, 255)][int]$Gpu = 0,
    [ValidateRange(1, 256)][int]$Jobs = 2,
    [string]$PythonExecutable = '',
    [string]$EnvironmentDirectory = '',
    [string]$BuildDirectory = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$workspace = Split-Path $repository -Parent

function Invoke-Native {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

function Get-WorkspacePath {
    param([string]$Path, [string]$Default)
    if (-not $Path) { $Path = $Default }
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $workspace $Path }
    return [IO.Path]::GetFullPath($Path)
}

function Test-SamePath {
    param([string]$First, [string]$Second)
    return [IO.Path]::GetFullPath($First).TrimEnd('\', '/') -ieq `
        [IO.Path]::GetFullPath($Second).TrimEnd('\', '/')
}

function Get-RequiredCommand {
    param([string]$Name)
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) { throw "Missing $Name. Install it and open a new PowerShell terminal." }
    return $command.Source
}

function Assert-SubmodulesClean {
    $states = @(Invoke-Native $git @('-C', $repository, 'submodule', 'status', '--recursive'))
    $missing = 0
    foreach ($state in $states) {
        if ($state -notmatch '^([ +\-U])[0-9a-f]+ (.+?)(?: \([^\r\n]*\))?$') {
            throw "Cannot parse submodule status: $state"
        }
        $status = $Matches[1]
        $relative = $Matches[2]
        if ($status -eq '-') { $missing++; continue }
        if ($status -ne ' ') {
            throw "Submodule '$relative' differs from the pinned commit or has a conflict. Preserve your work and resolve it before deployment; this script will not overwrite it."
        }
        $submodule = Join-Path $repository $relative
        $changes = @(Invoke-Native $git @('--no-optional-locks', '-C', $submodule, 'status', '--porcelain', '--untracked-files=normal'))
        if ($changes.Count -gt 0) {
            throw "Submodule '$relative' has local changes or untracked files. Commit, back up, or otherwise resolve them before deployment."
        }
    }
    Write-Host "Submodules: clean at their pinned commits; $missing not initialized."
}

function Assert-Variants {
    $configuration = Join-Path $build 'mitsuba.conf'
    if (Test-Path -LiteralPath $configuration) {
        # The generated configuration is JSON with whole-line # comments.
        $json = (Get-Content -LiteralPath $configuration -Encoding UTF8 | Where-Object { $_ -notmatch '^\s*#' }) -join "`n"
        $variants = @((ConvertFrom-Json $json).enabled)
        foreach ($required in @('scalar_rgb', 'cuda_ad_rgb')) {
            if ($required -notin $variants) {
                throw "$configuration must enable '$required'. Edit its enabled list or choose a new -BuildDirectory; MI_DEFAULT_VARIANTS does not update an existing file."
            }
        }
        Write-Host "Existing variants: $($variants -join ', ')"
    }
}

if ($env:OS -ne 'Windows_NT') { throw 'This script requires Windows.' }
$environment = Get-WorkspacePath $EnvironmentDirectory 'envs\mitsuba'
$build = Get-WorkspacePath $BuildDirectory (Join-Path $repository 'build')
$python = Join-Path $environment 'Scripts\python.exe'
if (Test-SamePath $build $repository) { throw 'The build directory must differ from the source directory.' }
Write-Host "Source: $repository"
Write-Host "Environment: $environment"
Write-Host "Build: $build"

$git = Get-RequiredCommand 'git.exe'
$cmake = Get-RequiredCommand 'cmake.exe'
$nvidia = Get-RequiredCommand 'nvidia-smi.exe'
Invoke-Native $git @('--version')
Invoke-Native $cmake @('--version')
Write-Host 'Selected GPU (index, name, driver, memory MiB):'
Invoke-Native $nvidia @("--id=$Gpu", '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader,nounits')

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio 2022 or Build Tools 2022 with Desktop development with C++ and a Windows SDK (vswhere.exe is missing).'
}
$visualStudio = @(Invoke-Native $vswhere @('-latest', '-products', '*', '-version', '[17.0,18.0)', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'))
if ($visualStudio.Count -ne 1) { throw 'Visual Studio 2022 C++ x64 tools were not found. Install the Desktop development with C++ workload.' }
$visualStudio = $visualStudio[0].Trim()
$toolVersionFile = Join-Path $visualStudio 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
$toolVersion = (Get-Content -LiteralPath $toolVersionFile -Raw).Trim()
$compiler = Join-Path $visualStudio "VC\Tools\MSVC\$toolVersion\bin\Hostx64\x64\cl.exe"
if (-not (Test-Path -LiteralPath $compiler)) { throw "MSVC x64 compiler is missing: $compiler" }
$sdkRoot = $null
foreach ($registryPath in @('HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots', 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots')) {
    $sdkRegistry = Get-ItemProperty -LiteralPath $registryPath -ErrorAction SilentlyContinue
    if ($sdkRegistry -and $sdkRegistry.PSObject.Properties['KitsRoot10']) { $sdkRoot = $sdkRegistry.KitsRoot10; break }
}
if (-not $sdkRoot) { throw 'Install a Windows 10 or Windows 11 SDK using Visual Studio Installer.' }
$sdk = @(Get-ChildItem -LiteralPath (Join-Path $sdkRoot 'Include') -Directory | Where-Object {
    (Test-Path -LiteralPath (Join-Path $_.FullName 'um\Windows.h')) -and
    (Test-Path -LiteralPath (Join-Path $_.FullName 'ucrt\stdio.h')) -and
    (Test-Path -LiteralPath (Join-Path $sdkRoot "Lib\$($_.Name)\um\x64\kernel32.lib"))
})
if ($sdk.Count -eq 0) { throw 'A complete Windows SDK with x64 libraries was not found. Repair it in Visual Studio Installer.' }
Write-Host "Visual Studio: $visualStudio (MSVC $toolVersion)"
Write-Host "Windows SDK: $(($sdk | Select-Object -Last 1).Name)"

$pythonCheck = "import json,struct,sys; assert sys.version_info[:2] == (3,12), sys.version; assert struct.calcsize('P') == 8, '64-bit Python is required'; print(json.dumps({'executable':sys.executable,'version':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
$creator = $null
if (Test-Path -LiteralPath $environment) {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "$environment exists but is not a usable Windows venv. Choose another -EnvironmentDirectory; existing directories are not replaced."
    }
    $info = ConvertFrom-Json ((Invoke-Native $python @('-I', '-B', '-c', $pythonCheck)) -join "`n")
    if ((Test-SamePath $info.prefix $info.base_prefix) -or -not (Test-SamePath $info.prefix $environment)) {
        throw 'The selected Python is not an independent venv at the requested environment path.'
    }
    $venvConfiguration = Join-Path $environment 'pyvenv.cfg'
    if (-not (Test-Path -LiteralPath $venvConfiguration) -or
        -not (Select-String -LiteralPath $venvConfiguration -Pattern '^include-system-site-packages\s*=\s*false\s*$' -Quiet)) {
        throw 'The venv must disable system site-packages. Create an isolated environment in another directory.'
    }
    Write-Host "Existing Python: $($info.version)"
    Invoke-Native $python @('-I', '-B', '-m', 'pip', '--version')
    $numpyCheck = "import importlib.metadata as m; print('NumPy: ' + next((d.version for d in m.distributions() if d.metadata['Name'].lower() == 'numpy'), 'not installed (deployment will install it)'))"
    Invoke-Native $python @('-I', '-B', '-c', $numpyCheck)
} else {
    if ($PythonExecutable) {
        $creator = Get-WorkspacePath $PythonExecutable $PythonExecutable
        $creatorArguments = @()
    } else {
        $creator = Get-RequiredCommand 'py.exe'
        $creatorArguments = @('-3.12')
    }
    $info = ConvertFrom-Json ((Invoke-Native $creator ($creatorArguments + @('-I', '-B', '-c', $pythonCheck))) -join "`n")
    Write-Host "New venv will use: $($info.executable)"
}

$cachePath = Join-Path $build 'CMakeCache.txt'
if (Test-Path -LiteralPath $cachePath) {
    $cache = @{}
    foreach ($line in Get-Content -LiteralPath $cachePath -Encoding UTF8) {
        if ($line -match '^([^#/:][^:]*):[^=]+=(.*)$') { $cache[$Matches[1]] = $Matches[2] }
    }
    foreach ($pair in @(@('CMAKE_HOME_DIRECTORY', $repository), @('CMAKE_CACHEFILE_DIR', $build), @('Python_EXECUTABLE', $python))) {
        if (-not $cache.ContainsKey($pair[0]) -or -not (Test-SamePath $cache[$pair[0]] $pair[1])) {
            throw "Existing CMake cache has a different or missing $($pair[0]). Use a new -BuildDirectory; the old build will not be overwritten."
        }
    }
    if ($cache['CMAKE_GENERATOR'] -ne 'Visual Studio 17 2022' -or $cache['CMAKE_GENERATOR_PLATFORM'] -ne 'x64') {
        throw 'Existing build is not Visual Studio 2022 x64. Choose a new -BuildDirectory.'
    }
    if ($cache.ContainsKey('CMAKE_GENERATOR_INSTANCE') -and $cache['CMAKE_GENERATOR_INSTANCE'] -and
        -not (Test-SamePath $cache['CMAKE_GENERATOR_INSTANCE'] $visualStudio)) {
        throw 'Existing build uses another Visual Studio installation. Choose a new -BuildDirectory.'
    }
    if ($cache.ContainsKey('_Python_EXECUTABLE') -and -not (Test-SamePath $cache['_Python_EXECUTABLE'] $python)) {
        throw 'Existing build cached another Python interpreter. Choose a new -BuildDirectory.'
    }
    Write-Host 'Existing CMake cache matches this source, environment, and generator.'
}
Assert-Variants
Assert-SubmodulesClean
Invoke-Native $git @('-C', $repository, 'rev-parse', 'HEAD')

if ($CheckOnly) {
    Write-Host 'Preflight passed. No environment, dependencies, submodules, or build files were changed.'
    Write-Host 'Run this script without -CheckOnly to build and run the CUDA and custom RGB-only checks.'
    return
}

$runName = 'windows-deploy-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$output = Join-Path $workspace "outputs\mitsuba\$runName"
New-Item -ItemType Directory -Path $output -ErrorAction Stop | Out-Null
$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath (Join-Path $output 'deployment.log') | Out-Null
    $transcriptStarted = $true
    if ($creator) {
        Invoke-Native $creator ($creatorArguments + @('-I', '-B', '-m', 'venv', $environment))
    }
    Invoke-Native $python @('-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install', 'numpy')
    Invoke-Native $git @('-C', $repository, 'submodule', 'update', '--init', '--recursive')
    Assert-SubmodulesClean
    Invoke-Native $cmake @('-S', $repository, '-B', $build, '-G', 'Visual Studio 17 2022', '-A', 'x64',
        "-DCMAKE_GENERATOR_INSTANCE=$visualStudio", "-DPython_EXECUTABLE=$python", '-DMI_DEFAULT_VARIANTS=scalar_rgb,cuda_ad_rgb',
        '-DMI_SPLIT_MODE=OFF', '-DMI_ENABLE_EMBREE=OFF', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5')
    Assert-Variants
    Invoke-Native $cmake @('--build', $build, '--config', 'Release', '--parallel', "$Jobs")
    $runtime = Join-Path $build 'Release'
    Invoke-Native $python @('-I', '-B', (Join-Path $PSScriptRoot 'smoke_gpu.py'),
        '--build-dir', $runtime, '--output-dir', (Join-Path $output 'source-smoke'), '--gpu', "$Gpu")
    Invoke-Native $python @('-I', '-B', (Join-Path $PSScriptRoot '..\tests\test_reconstruction_rgb_only.py'),
        '--build-dir', $runtime, '--gpu', "$Gpu")
    Write-Host "Deployment checks passed. Reports and log: $output"
    Write-Host "Training interpreter: $python"
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
