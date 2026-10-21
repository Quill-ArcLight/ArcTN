"""Select idle NUMA-local CPU slots and share the server's existing core locks."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(os.environ.get('QC32_SCHEDULER_ROOT', '.'))


def cpu_list(value):
    result = set()
    for item in value.strip().split(','):
        span = item.split('-')
        result.update(range(int(span[0]), int(span[-1]) + 1))
    return result


def topology(cpu):
    path = Path(f'/sys/devices/system/cpu/cpu{cpu}')
    node = next(path.glob('node[0-9]*'))
    top = path / 'topology'
    return (int(node.name[4:]), int((top / 'physical_package_id').read_text()),
            int((top / 'core_id').read_text()), cpu_list((top / 'thread_siblings_list').read_text()))


def lock_paths(cpus):
    paths = set()
    for cpu in cpus:
        _, socket, core, _ = topology(cpu)
        paths.add(ROOT / 'scheduler/physical-core-locks' / f'cpu-{cpu}.lock')
        for directory in ('locks/physical-cores', 'slot-locks-v1'):
            paths.add(ROOT / directory / f'socket{socket}-core{core}.lock')
    return sorted(paths)


def acquire_locks(cpus):
    fds = []
    try:
        for path in lock_paths(cpus):
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o660)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(fd)
                raise
            fds.append(fd)
        return fds
    except BaseException:
        release_locks(fds)
        raise


def release_locks(fds):
    for fd in fds:
        os.close(fd)


def cpu_times():
    values = {}
    for line in Path('/proc/stat').read_text().splitlines():
        name, *fields = line.split()
        if name.startswith('cpu') and name[3:].isdigit():
            v = list(map(int, fields[:8]))
            values[int(name[3:])] = (sum(v), v[3] + v[4])
    return values


def choose_slots():
    start = cpu_times()
    time.sleep(2)
    end = cpu_times()
    busy = {c: 100 * (1 - (end[c][1] - start[c][1]) / max(1, end[c][0] - start[c][0])) for c in end}
    pinned = set()
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            if 'VmSize' not in fields:
                continue
            affinity = cpu_list(fields['Cpus_allowed_list'])
            if len(affinity) <= 64:
                pinned.update(affinity)
        except (OSError, KeyError, ValueError):
            continue
    nodes = {}
    for cpu in sorted(end):
        node, _, _, siblings = topology(cpu)
        if cpu == min(siblings):
            nodes.setdefault(node, []).append((cpu, siblings))
    eligible = []
    for node, rows in nodes.items():
        cpus = set().union(*(s for _, s in rows))
        if len(rows) != 16 or cpus & pinned or max(busy[c] for c in cpus) > 10:
            continue
        try:
            fds = acquire_locks(cpus)
            release_locks(fds)
        except OSError:
            continue
        eligible.append((sum(busy[c] for c in cpus), node, rows))
    eligible.sort()
    if len(eligible) < 4:
        raise RuntimeError(f'Only {len(eligible)} complete idle NUMA nodes available; need four for 64 cores')
    chosen = eligible[:4]
    slots = []
    reserved = set()
    for _, _, rows in chosen:
        cpus = sorted(c for c, _ in rows)
        slots.extend((cpus[:8], cpus[8:]))
        reserved.update(set().union(*(s for _, s in rows)))
    control = None
    for cpu in sorted(end):
        _, _, _, siblings = topology(cpu)
        if cpu != min(siblings) or siblings & (reserved | pinned) or max(busy[c] for c in siblings) > 10:
            continue
        try:
            fds = acquire_locks(siblings)
            release_locks(fds)
            control = cpu
            reserved.update(siblings)
            break
        except OSError:
            continue
    if control is None:
        raise RuntimeError('No separate idle controller core is available')
    return {'schema': 'qc32-live-physical-slots-v1', 'live_gate_status': 'passed',
            'checked_utc': datetime.now(timezone.utc).isoformat(), 'slots': slots,
            'controller_cpu': control, 'reserved_logical_cpus': sorted(reserved),
            'numa_nodes': [n for _, n, _ in chosen],
            'sampled_cpu_busy_percent': {str(c): round(busy[c], 3) for c in sorted(reserved)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', type=Path, default=Path('slots.json'))
    parser.add_argument('--build', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    slots = choose_slots()
    args.slots.write_text(json.dumps(slots, indent=2) + '\n')
    print(json.dumps({k: slots[k] for k in ('slots', 'controller_cpu', 'numa_nodes')}), flush=True)
    if args.build:
        cpus = slots['slots'][0]
        reserved = set().union(*(topology(c)[3] for c in cpus))
        fds = acquire_locks(reserved)
        try:
            raise SystemExit(subprocess.call(['taskset', '--cpu-list', ','.join(map(str, cpus)), *args.build]))
        finally:
            release_locks(fds)


if __name__ == '__main__':
    main()
