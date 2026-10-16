"""Exclusive KV ownership for one independently scheduled decoder replica."""
from dataclasses import dataclass


@dataclass
class ReplicaLease:
    state: str = 'idle'
    request_id: str | None = None

    def begin_admission(self, request_id):
        if self.state != 'idle' or not request_id:
            raise RuntimeError('KV admission requires an idle replica and request ID')
        self.state, self.request_id = 'admitting', request_id

    def mark_ready(self, completed_cqes, expected_cqes):
        if (self.state != 'admitting' or expected_cqes <= 0 or
                completed_cqes != expected_cqes):
            raise RuntimeError('All DMA completions are required before decode')
        self.state = 'ready'

    def start_decode(self):
        if self.state != 'ready':
            raise RuntimeError('Decode requires completed KV admission')
        self.state = 'decoding'

    def finish_decode(self, request_id, passed):
        if (self.state != 'decoding' or request_id != self.request_id or
                passed is not True):
            raise RuntimeError('Only the matching successful completion releases KV')
        self.state, self.request_id = 'idle', None
