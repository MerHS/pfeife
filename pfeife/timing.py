from collections import defaultdict
import time

import torch


class CPUEvent:
    def __init__(self):
        self.time = time.time()

    def elapsed_time(self, end):
        return end.time - self.time


class TimeRecorder:
    def __init__(self):
        self.fw_events = defaultdict(list)
        self.bw_events = defaultdict(list)

    def reset(self):
        self.fw_events.clear()
        self.bw_events.clear()

    def get_event_logs(self):
        fw_time = defaultdict(list)
        bw_time = defaultdict(list)

        for key, times in self.fw_events.items():
            for start, end in times:
                fw_time[key].append(start.elapsed_time(end))
        for key, times in self.bw_events.items():
            for start, end in times:
                bw_time[key].append(start.elapsed_time(end))

        return fw_time, bw_time

    def append_fw(self, curr_id, start_event, end_event):
        self.fw_events[curr_id].append((start_event, end_event))

    def append_bw(self, curr_id, start_event, end_event):
        self.bw_events[curr_id].append((start_event, end_event))


class CPUTimeRecorder(TimeRecorder):
    def record_event(self, device, stream=None):
        return CPUEvent()


class CUDATimeRecorder(TimeRecorder):
    def record_event(self, device, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream(device)

        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event
