#! /usr/bin/env bash
help=$(cat << EOF
Container entrypoint script.
Arguments:
  --help: Print this help message.
  --start: Start the container instance.
  --stop: Stop the container instance.
  --tmux <session_name>: Attach to a tmux session inside the container instance. If the session doesn't exist, it will be created.
  --exec <command>: Execute a command inside the container instance.
  --refresh-tmux <pid>: Send a USR1 signal to the given process ID inside the container instance. This can be used to refresh tmux sessions.
  --ps: List all processes running inside the container instance.
EOF
)

ENVS=()
ENVS+=("APPTAINERENV_CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11")
ENVS+=(TMPDIR=/home/user/.tmp)
ENVS+=(TEMP=/home/user/.tmp)
ENVS+=(TMP=/home/user/.tmp)
ENVS+=(TMUX_TMPDIR=/home/user/.tmp)
ENVS+=(PYTORCH_ALLOC_CONF=expandable_segments:True)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
        echo "$help"
        exit 0
        ;;
    --start)
        env "${ENVS[@]}" apptainer instance start train.sif train-node
        exit $?
        ;;
    --stop)
        apptainer instance stop train-node
        exit $?
        ;;
    --tmux)
        # try to attach to existing tmux session, if it doesn't exist, create a new one
        name="$2"
        shift
        shift
        if ! env "${ENVS[@]}" apptainer exec instance://train-node tmux attach-session -t "$name" "$@" 2>/dev/null; then
            env "${ENVS[@]}" apptainer exec instance://train-node tmux new -s "$name" "$@"
        fi
        exit $?
        ;;
    --exec)
        shift
        env "${ENVS[@]}" apptainer exec instance://train-node "$@"
        exit $?
        ;;
    --refresh-tmux)
       shift
       env "${ENVS[@]}" apptainer exec instance://train-node kill -USR1 $1
       exit $?
       ;;
    --ps)
       env "${ENVS[@]}" apptainer exec instance://train-node ps -e
       exit $?
       ;;
    *)
        POSITIONAL_ARGS+=("$1") # save positional arg
        shift # past argument
        ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "$help"