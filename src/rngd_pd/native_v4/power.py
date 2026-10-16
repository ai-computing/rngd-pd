"""Independent board-power samplers; timestamps bracket each sensor command."""
import json
import subprocess
import threading
import time


class PowerSampler:
    def __init__(self, output, rngd_device, atom_device, interval=.5):
        self.output = output
        self.devices = {'rngd': rngd_device}
        if isinstance(atom_device, list):
            self.devices.update({f'atom-{device}': device for device in atom_device})
        else:
            self.devices['atom'] = atom_device
        self.interval = interval
        self.stop = threading.Event()
        self.ready = {kind: threading.Event() for kind in self.devices}
        self.latest = {kind: 0. for kind in self.devices}
        self.threads = []

    def _sample(self, kind):
        with (self.output/f'power-{kind}.jsonl').open('a') as out:
            while not self.stop.is_set():
                start = time.perf_counter()
                try:
                    if kind == 'rngd':
                        data = json.loads(subprocess.check_output(
                            ['furiosa-smi', 'info', '--format', 'json'], timeout=5))
                        device = next(d for d in data if d['dev_name'] == self.devices[kind])
                        watts = float(device['power'].split()[0])
                    else:
                        data = json.loads(subprocess.check_output(['rbln-smi', '--json'], timeout=5))
                        device = next(d for d in data['devices'] if d['npu'] == self.devices[kind])
                        watts = float(device['card_power'].removesuffix('uW'))/1e6
                    end = time.perf_counter()
                    row = dict(start=start, end=end, watts=watts, device=device)
                    self.latest[kind] = (start+end)/2
                    self.ready[kind].set()
                except Exception as exc:
                    row = dict(start=start, end=time.perf_counter(), error=repr(exc))
                out.write(json.dumps(row)+'\n');out.flush()
                self.stop.wait(self.interval)

    def start(self):
        for kind in self.devices:
            thread = threading.Thread(target=self._sample, args=(kind,), daemon=True)
            thread.start();self.threads.append(thread)
        if not all(event.wait(10) for event in self.ready.values()):
            raise RuntimeError('power sensor initialization failed')

    def finish(self, window_end=None):
        deadline = time.monotonic()+10
        while window_end is not None and min(self.latest.values()) < window_end:
            if time.monotonic()>deadline:
                break  # The analysis rejects an unbracketed window.
            time.sleep(.05)
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=6)
