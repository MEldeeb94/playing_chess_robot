:: Copyright (c) 2018 Roboception GmbH
:: All rights reserved
::
:: Author: Raphael Schaller
::
:: Redistribution and use in source and binary forms, with or without
:: modification, are permitted provided that the following conditions are met:
::
:: 1. Redistributions of source code must retain the above copyright notice,
:: this list of conditions and the following disclaimer.
::
:: 2. Redistributions in binary form must reproduce the above copyright notice,
:: this list of conditions and the following disclaimer in the documentation
:: and/or other materials provided with the distribution.
::
:: 3. Neither the name of the copyright holder nor the names of its contributors
:: may be used to endorse or promote products derived from this software without
:: specific prior written permission.
::
:: THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
:: AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
:: IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
:: ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
:: LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
:: CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
:: SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
:: INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
:: CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
:: ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
:: POSSIBILITY OF SUCH DAMAGE.

@echo off
setlocal enableDelayedExpansion

set "SCRIPT_PATH=%~dp0"

:: commands
set "GC_FOLDER=!SCRIPT_PATH!\rc_genicam_api-win32-vc15-v2.2.3\bin"
set "DYN_FOLDER=!SCRIPT_PATH!\rc_dynamics_api-win32-vc14-v0.7.0\bin"
set "GC_STREAM=!GC_FOLDER!\gc_stream"
set "GC_INFO=!GC_FOLDER!\gc_info"
set "GC_CONFIG=!GC_FOLDER!\gc_config"
set "RCDISCOVER=!SCRIPT_PATH!\rcdiscover\rcdiscover.exe"
set "CURL=!DYN_FOLDER!\curl.exe"
set "DYN_STREAM=!DYN_FOLDER!\rcdynamics_stream.exe"

set "SCRIPT=%~0"

:: rc_visard serial number
set "SN="

:: default parameters
set "LEFT=true"
set "RIGHT=true"
set "DISPARITY=true"
set "CONFIDENCE=true"
set "ERROR=true"

set "CAM_PARAMS="
set "SLAM="
set "IMU="
set "HAND_EYE_CALIB="
set "PROJECTOR="

set "NUMBER=1000000"
set "FRAME_RATE=3"

set "VERBOSE=false"
set "YES=false"
set "MONO=false"

set "OUT_FOLDER=rc-visard_$S_$T"

(set \n=^
%=DONT REMOVE THIS=%
)

:: Usage text
set "USAGE=Usage: !SCRIPT! [options] <rc_visard serial number>!\n! Options!\n!  -h: print this message!\n!  -v: enable verbose mode!\n!  -y: assume Yes to all queries and do not prompt!\n!  -m: record monochrome images even if color camera!\n!  -n [int]: number of images to capture!\n!  -o [string] (default: %OUT_FOLDER%): if non-empty, store data in given folder. $S will be replaced by the serial number, $T by the current timestamp.!\n!  -f [int] (default: %FRAME_RATE%): capturing frame rate!\n!  -left [true,false] (default: %LEFT%): capture left image!\n!  -right [true,false] (default: %RIGHT%): capture right image!\n!  -disparity [true,false] (default: %DISPARITY%): capture disparity image!\n!  -confidence [true,false] (default: %CONFIDENCE%): capture confidence image!\n!  -error [true,false] (default: %ERROR%): capture error image!\n!  -cam-params [string]: write camera parameters to given file!\n!  -slam [string]: enable SLAM and write trajectory to given file!\n!  -imu [string]: record some IMU samples and write them to the given file!\n!  -hand-eye-calib [string]: write hand-eye-calibration to given file!\n!  -projector [Low,High,ExposureActive,ExposureAlternateActive]: projector mode. If not set, projector mode is not changed.

if "%1"=="" (
  set NO_ARGS_PASSED=y
)

:loop
if not "%1"=="" (
  set ARG=%1
  if "!ARG!"=="-h" (
    set PRINT_HELP=true
  ) else (
  if "!ARG!"=="-v" (
    set VERBOSE=true
  ) else (
  if "!ARG!"=="-y" (
    set YES=true
  ) else (
  if "!ARG!"=="-m" (
    set MONO=true
  ) else (
  if "!ARG!"=="-n" (
    set "NUMBER=%2"
    call :check_is_number !NUMBER! , "-n"
    shift
  ) else (
  if "!ARG!"=="-o" (
    set "OUT_FOLDER=%2"
    shift
  ) else (
  if "!ARG!"=="-f" (
    set "FRAME_RATE=%2"
    call :check_is_number !FRAME_RATE! , "-f"
    shift
  ) else (
  if "!ARG!"=="-left" (
    set "LEFT=%2"
    call :check_is_bool !LEFT! , "-left"
    shift
  ) else (
  if "!ARG!"=="-right" (
    set "RIGHT=%2"
    call :check_is_bool !RIGHT! , "-right"
    shift
  ) else (
  if "!ARG!"=="-disparity" (
    set "DISPARITY=%2"
    call :check_is_bool !DISPARITY! , "-disparity"
    shift
  ) else (
  if "!ARG!"=="-confidence" (
    set "CONFIDENCE=%2"
    call :check_is_bool !CONFIDENCE! , "-confidence"
    shift
  ) else (
  if "!ARG!"=="-error" (
    set "ERROR=%2"
    call :check_is_bool !ERROR! , "-error"
    shift
  ) else (
  if "!ARG!"=="-cam-params" (
    set "CAM_PARAMS=%2"
    shift
  ) else (
  if "!ARG!"=="-slam" (
    set "SLAM=%2"
    shift
  ) else (
  if "!ARG!"=="-imu" (
    set "IMU=%2"
    shift
  ) else (
  if "!ARG!"=="-hand-eye-calib" (
    set "HAND_EYE_CALIB=%2"
    shift
  ) else (
  if "!ARG!"=="-projector" (
    set "PROJECTOR=%2"
    shift
  ) else (
  if "!ARG:~0,1!"=="-" (
    echo Option '!ARG!' not known
    set PARSE_ERROR=y
  ) )))))))))))))))))
  shift
  if not defined PARSE_ERROR (
    goto :loop
  )
)

if defined NO_ARGS_PASSED (
  if "!SN!"=="" (
    :interactive_loop
    set /P "c=Enter the rc_visard's serial number or user defined name, 'L' to list available rc_visards, or 'h' to print the help text "
    if "!c!"=="h" (
      set PRINT_HELP=y
    )
    if /I "!c!"=="l" (
      call "!GC_CONFIG!" -l
      goto :interactive_loop
    )
    set SN=!c!
  )
)
if not defined NO_ARGS_PASSED (
  if not "%0"=="" (
    set SN=%0
  )
)

if not defined PARSE_ERROR (
  if defined PRINT_HELP (
    echo !USAGE!
    if defined NO_ARGS_PASSED ( pause )
    goto :eof
  )
)

if defined PARSE_ERROR (
  echo !USAGE!
  if defined NO_ARGS_PASSED ( pause )
  goto :eof
)

set TS=%date:~-4%-%date:~-7,2%-%date:~-10,2%_%time:~-11,2%-%time:~-8,2%-%time:~-5,2%
set TS=!TS:^ =0!
set OUT_FOLDER=%OUT_FOLDER:$T=!TS!%
set OUT_FOLDER=%OUT_FOLDER:$S=!SN!%

call :check_overwrite_folder !OUT_FOLDER!
if defined CANCEL_OVERWRITE (
  goto :eof
)

if not "!OUT_FOLDER!"=="" (
  if not exist "!OUT_FOLDER!" ( mkdir "!OUT_FOLDER!" )
  cd "!OUT_FOLDER!"
  echo Writing data to !cd!
)

call :check_overwrite_file !CAM_PARAMS!
call :check_overwrite_file !SLAM!
call :check_overwrite_file !IMU!
call :check_overwrite_file !HAND_EYE_CALIB!
if defined CANCEL_OVERWRITE (
  goto :eof
)

if not "!SLAM!"=="" ( set GET_IP=y )
if not "!IMU!"=="" ( set GET_IP=y )
if not "!HAND_EYE_CALIB!"=="" ( set GET_IP=y )

if defined GET_IP (
  for /F "tokens=*" %%g in ('call "!GC_CONFIG!" !SN! --iponly') do (set IP=%%g)

  if "!IP!"=="" (
    echo Sensor not found
    goto :eof
  )

  if "!VERBOSE!"=="true" ( echo Sensor IP is !IP! )
)

if not "!SLAM!"=="" (
  echo Starting SLAM ...
  call :conditionally_silence_cmd !CURL! -s -S -X PUT --header "Content-Type: application/json" --header "Accept: application/json" -d "{ }" "http://!IP!/api/v1/nodes/rc_dynamics/services/restart_slam"
)

if not "!CAM_PARAMS!"=="" (
  echo Writing camera parameters to '!CAM_PARAMS!'
  call "!GC_INFO!" !SN! > !CAM_PARAMS!
)

if not "!IMU!"=="" (
  echo Writing IMU samples to '!IMU!'
  call "!DYN_STREAM!" -v !IP! -s imu -o !IMU! -n 20
)

if not "!HAND_EYE_CALIB!"=="" (
  echo Writing hand-eye-calibration to '!HAND_EYE_CALIB!'
  call :conditionally_silence_cmd !CURL! -s -S -X PUT --header "Content-Type: application/json" --header "Accept: application/json" -d "{ }" "http://!IP!/api/v1/nodes/rc_hand_eye_calibration/services/get_calibration" -o !HAND_EYE_CALIB!
)

if "!MONO!"=="true" (
  set "COLOR_CMD=^"!GC_CONFIG!^" !SN! ComponentSelector=Intensity PixelFormat=Mono8 ComponentSelector=IntensityCombined PixelFormat=Mono8"
) else (
  set "COLOR_CMD=^"!GC_CONFIG!^" !SN! ComponentSelector=Intensity PixelFormat=YCbCr411_8 ComponentSelector=IntensityCombined PixelFormat=YCbCr411_8"
)

if "!VERBOSE!"=="true" ( echo !COLOR_CMD! )
call !COLOR_CMD! 2> nul

set "ALTERNATE_FILTER_OFF_CMD=^"!GC_CONFIG!^" !SN! AcquisitionAlternateFilter=Off"
if "!VERBOSE!"=="true" ( echo !ALTERNATE_FILTER_OFF_CMD! )
call !ALTERNATE_FILTER_OFF_CMD! 2> nul

set ENABLE_INTENSITY=0
set ENABLE_COMBINED=0
set ENABLE_DISPARITY=0
set ENABLE_CONFIDENCE=0
set ENABLE_ERROR=0

if "!LEFT!"=="true" (
  if "!RIGHT!"=="true" (
    set ENABLE_COMBINED=1
))
if "!LEFT!"=="false" (
  if "!RIGHT!"=="true" (
    set ENABLE_COMBINED=1
))
if "!LEFT!"=="true" (
  if "!RIGHT!"=="false" (
    set ENABLE_INTENSITY=1
))

if "!DISPARITY!"=="true" ( set ENABLE_DISPARITY=1 )
if "!CONFIDENCE!"=="true" ( set ENABLE_CONFIDENCE=1 )
if "!ERROR!"=="true" ( set ENABLE_ERROR=1 )

set "GC_COMMAND=^"!GC_STREAM!^" !SN! n=!NUMBER! AcquisitionFrameRate=!FRAME_RATE! ComponentSelector=Intensity ComponentEnable=!ENABLE_INTENSITY! ComponentSelector=IntensityCombined ComponentEnable=!ENABLE_COMBINED! ComponentSelector=Disparity ComponentEnable=!ENABLE_DISPARITY! ComponentSelector=Confidence ComponentEnable=!ENABLE_CONFIDENCE! ComponentSelector=Error ComponentEnable=!ENABLE_ERROR!"

if not "!PROJECTOR!"=="" (
  set "GC_COMMAND=!GC_COMMAND! LineSelector=Out1 LineSource=!PROJECTOR!"
)

if "!VERBOSE!"=="true" ( echo !GC_COMMAND! )
echo Start streaming ...

call !GC_COMMAND!

if not "!SLAM!"=="" (
  echo Writing trajectory to '!SLAM!' ...
  call :conditionally_silence_cmd !CURL! -s -S -X PUT --header "Content-Type: application/json" --header "Accept: application/json" -d "{ }" "http://%IP%/api/v1/nodes/rc_dynamics/services/stop"
  call :conditionally_silence_cmd !CURL! -s -S -X PUT --header "Content-Type: application/json" --header "Accept: application/json" -d "{ }" "http://%IP%/api/v1/nodes/rc_slam/services/get_trajectory" -o !SLAM!
)

if defined NO_ARGS_PASSED ( pause )
goto :eof

:check_is_number
set "var="&for /f "delims=0123456789" %%i in ("%~1") do set var=%%i
if defined var (
  echo %~2 is not a number
  set PARSE_ERROR=true
)
exit /B

:check_is_bool
set ok=false
if "%~1"=="true" (set ok=true)
if "%~1"=="false" (set ok=true)
if "!ok!"=="false" (
  echo %~2 is not a boolean
  set PARSE_ERROR=true
)
exit /B

:conditionally_silence_cmd
set CMD=%*
if "!VERBOSE!"=="true" (
  call !CMD!
) else (
  call !CMD! > nul
)
exit /B

:query_yes_no
set QUERY=%~1
set /P "c=!QUERY!"
set YES=
set NO=
if "!c!"=="y" (
  set YES=y
)
if "!c!"=="Y" (
  set YES=y
)
if not defined YES (
  set NO=y
)
exit /B

:check_overwrite_file
set FILE=%~1
if not "!FILE!"=="" (
  if exist !FILE! (
    call :query_yes_no "File '!FILE!' already exists. Are you sure to overwrite it? (y/N) "
    if defined NO ( set CANCEL_OVERWRITE=y )
  )
)
exit /B

:check_overwrite_folder
set FOLDER=%~1
if not "!FOLDER!"=="" (
  if exist !FOLDER! (
    call :query_yes_no "Folder '!FOLDER!' already exists. Are you sure use it? (y/N) "
    if defined NO ( set CANCEL_OVERWRITE=y )
  )
)
exit /B
