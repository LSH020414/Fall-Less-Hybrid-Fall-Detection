import argparse
import getpass
import os
import shlex
import sys
import time
from pathlib import Path, PurePosixPath


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "vendor"))

import paramiko


LOCAL_FILE = SCRIPT_DIR / "real_time_fall_detection.py"
LOCAL_STATUS_SCRIPT = SCRIPT_DIR / "fall_detection_live_status.sh"
LOCAL_STATUS_DESKTOP = SCRIPT_DIR / "Fall-Detection-Live-Status.desktop"
SERVICE_ENV = (
    "export XDG_RUNTIME_DIR=/run/user/$(id -u); "
    "export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus; "
)


def run(ssh, command, check=True):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=60)
    status = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if check and status != 0:
        raise RuntimeError(
            f"Command failed ({status}): {command}\n{output}\n{error}"
        )
    return status, output, error


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deploy Raspberry Pi TCN + FPGA GRU integration and "
            "decision-pipeline status display."
        )
    )
    parser.add_argument("--host", default=os.environ.get("PI_HOST", "192.168.137.2"))
    parser.add_argument("--user", default=os.environ.get("PI_USER", "raspberrylee"))
    parser.add_argument("--password", default=os.environ.get("PI_PASSWORD"))
    parser.add_argument(
        "--remote-dir",
        default=os.environ.get("PI_PROJECT_DIR", "/home/pi/fall_detection"),
    )
    args = parser.parse_args()
    remote_file = f"{args.remote_dir}/real_time_fall_detection.py"
    remote_home = str(PurePosixPath(args.remote_dir).parent)
    remote_status_script = f"{args.remote_dir}/fall_detection_live_status.sh"
    remote_status_desktop = (
        f"{remote_home}/Desktop/Fall-Detection-Live-Status.desktop"
    )

    if not LOCAL_FILE.exists():
        raise SystemExit(f"Missing local file: {LOCAL_FILE}")
    if not LOCAL_STATUS_SCRIPT.exists():
        raise SystemExit(f"Missing local file: {LOCAL_STATUS_SCRIPT}")
    if not LOCAL_STATUS_DESKTOP.exists():
        raise SystemExit(f"Missing local file: {LOCAL_STATUS_DESKTOP}")

    password = args.password or getpass.getpass(
        f"Password for {args.user}@{args.host}: "
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{remote_file}.pre_gru_{timestamp}.bak"
    remote_temp = f"{remote_file}.uploading"

    ssh = None
    last_connect_error = None
    for attempt in range(1, 13):
        candidate = paramiko.SSHClient()
        candidate.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(
            f"Connecting to {args.user}@{args.host} "
            f"(attempt {attempt}/12) ..."
        )
        try:
            candidate.connect(
                args.host,
                username=args.user,
                password=password,
                timeout=8,
                banner_timeout=8,
                auth_timeout=8,
                look_for_keys=False,
                allow_agent=False,
            )
            ssh = candidate
            break
        except Exception as exc:
            last_connect_error = exc
            candidate.close()
            if attempt < 12:
                time.sleep(5)

    if ssh is None:
        raise RuntimeError(
            f"Could not connect to Raspberry Pi: {last_connect_error}"
        )

    backup_created = False
    try:
        run(
            ssh,
            SERVICE_ENV
            + "systemctl --user stop fall-detection.service || true",
        )
        run(
            ssh,
            f"test -f {shlex.quote(remote_file)} && "
            f"cp {shlex.quote(remote_file)} {shlex.quote(backup)}",
        )
        backup_created = True

        run(ssh, f"mkdir -p {shlex.quote(remote_home + '/Desktop')}")

        sftp = ssh.open_sftp()
        try:
            sftp.put(str(LOCAL_FILE), remote_temp)
            sftp.chmod(remote_temp, 0o644)
            sftp.put(str(LOCAL_STATUS_SCRIPT), remote_status_script)
            sftp.chmod(remote_status_script, 0o755)
            sftp.put(str(LOCAL_STATUS_DESKTOP), remote_status_desktop)
            sftp.chmod(remote_status_desktop, 0o755)
        finally:
            sftp.close()

        run(
            ssh,
            f"{args.remote_dir}/yolovenv/bin/python -m py_compile "
            f"{shlex.quote(remote_temp)}",
        )
        run(
            ssh,
            f"mv {shlex.quote(remote_temp)} {shlex.quote(remote_file)}",
        )
        run(
            ssh,
            SERVICE_ENV
            + "systemctl --user daemon-reload && "
            "systemctl --user restart fall-detection.service",
        )
        time.sleep(5)
        _, status_output, _ = run(
            ssh,
            SERVICE_ENV
            + "systemctl --user --no-pager --full status "
            "fall-detection.service",
            check=False,
        )
        _, journal_output, journal_error = run(
            ssh,
            SERVICE_ENV
            + "journalctl --user -u fall-detection.service "
            "--no-pager -n 40",
            check=False,
        )
        print(status_output)
        print(journal_output or journal_error)

        if "Active: active (running)" not in status_output:
            raise RuntimeError("Service did not reach active (running) state.")

        print("DEPLOYMENT_COMPLETE")
        print(f"Backup: {backup}")
    except Exception:
        if backup_created:
            print("Deployment failed. Restoring the previous Pi script.")
            run(
                ssh,
                f"cp {shlex.quote(backup)} {shlex.quote(remote_file)}",
                check=False,
            )
            run(
                ssh,
                SERVICE_ENV
                + "systemctl --user restart fall-detection.service",
                check=False,
            )
        raise
    finally:
        run(ssh, f"rm -f {shlex.quote(remote_temp)}", check=False)
        ssh.close()


if __name__ == "__main__":
    main()
