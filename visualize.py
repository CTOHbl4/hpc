import os
import signal
import sys

PIPE_FILE = "/tmp/viz_pipe"


def write_command(command):
    """Write a command to the pipe file"""
    try:
        with open(PIPE_FILE, 'w') as f:
            f.write(command + '\n')
        print(f"Sent: {command}")
    except Exception as e:
        print(f"Error writing: {e}")


def cleanup():
    """Clean up the pipe file"""
    if os.path.exists(PIPE_FILE):
        os.remove(PIPE_FILE)
        print(f"Removed pipe file: {PIPE_FILE}")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal")
    cleanup()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    if not os.path.exists(PIPE_FILE):
        open(PIPE_FILE, 'w').close()
        print(f"Created pipe file: {PIPE_FILE}")

    print("Controller started. Enter commands in format: x,y,z")
    print("Example: 32,16,48")
    print("Press Ctrl+C to remove the pipe and exit")

    try:
        while True:
            cmd = input("> ")
            if cmd.strip():
                write_command(cmd.strip())
    finally:
        cleanup()
        print("Controller stopped")


if __name__ == "__main__":
    main()
