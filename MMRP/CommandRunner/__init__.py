import threading
import logging
import platform
import subprocess
import psutil
OS_LINUX = "ELF"


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


class CommandRunner:
    def __init__(self, output_size: int = 500):
        self.log_size = output_size
        self.output = list()
        self.lastline = ''
        self.cmd = None
        _, os = platform.architecture()
        self.os = OS_LINUX if not os else os

    def log(self, log_line):
        print(log_line)

    def callback(self):
        raise NotImplementedError

    def run_cmd(self, cmd: str, timeout: int = -1) -> int:
        logging.info("Running cmd: \"{}\"".format(cmd))
        if self.os == OS_LINUX:
            self.cmd = cmd
        else:
            self.cmd = cmd + " & exit"
        self.process = subprocess.Popen(self.cmd, shell=True, bufsize=1024, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        log_thread = threading.Thread(
            target=self.print_log, args=(self.process.stdout,))
        log_thread.start()

        if timeout > 0:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill(self.process.pid)
                log_thread.join()
                raise
        else:
            self.process.wait()
        log_thread.join()

        # return self.callback()

    def print_log(self, stdout):
        for log_line in self._log_line_iter(stdout):
            self.log(log_line)
            if len(self.output) > self.log_size:
                del self.output[0]
            self.output.append(log_line)

    def _log_line_iter(self, reader):
        while True:
            # fix massive log(memory error)
            buf = reader.read(1024)
            if buf:
                if self.os == OS_LINUX:
                    lines = buf.decode('utf8', errors='ignore')
                else:
                    lines = buf.decode('utf8', errors='ignore')
                lines = lines.replace('\r\n', '\n').replace(
                    '\r', '\n').split('\n')
                lines[0] = self.lastline + lines[0]
                for line in lines[:-1]:
                    if len(line) > 0:
                        yield line
                self.lastline = lines[-1]
            else:
                break
