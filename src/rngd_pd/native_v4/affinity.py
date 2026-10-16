"""Process-local CPU placement; no global BIOS, driver, or kernel changes."""
from pathlib import Path


def parse_cpu_list(spec):
    cpus = set()
    for part in spec.strip().split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            cpus.update(range(lo, hi+1))
        else:
            cpus.add(int(part))
    return cpus


def pci_cpus(bdf, allowed):
    node = int((Path('/sys/bus/pci/devices')/bdf/'numa_node').read_text())
    if node < 0:
        raise RuntimeError(f'Unknown NUMA node for {bdf}')
    cpus = parse_cpu_list((Path('/sys/devices/system/node')/f'node{node}'/'cpulist').read_text()) & set(allowed)
    if not cpus:
        raise RuntimeError(f'No allowed CPUs on NUMA node {node}')
    return node, sorted(cpus)
