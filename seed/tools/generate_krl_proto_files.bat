@echo off

set PYTHON_OUT=../
set PROTO_INCLUDE=../
set PROTO_FILE_DIR=../krl/proto/

for %%i in (%PROTO_FILE_DIR%*.proto) do (
    echo Generating grpc code for: %PROTO_FILE_DIR%%%i
    python -m grpc_tools.protoc -I%PROTO_INCLUDE% --python_out=%PYTHON_OUT% --grpc_python_out=%PYTHON_OUT% %PROTO_FILE_DIR%%%i
)