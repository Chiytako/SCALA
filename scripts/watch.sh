#!/usr/bin/env bash
ssh -p 11854 root@ssh7.vast.ai "python3 /workspace/watch.py 8B-v2 2670 /workspace/runs/scala-8b-a1b-v2/log.jsonl" 2>/dev/null | tail -1
ssh sit@100.64.3.0 "python3 /tmp/watch.py v2-A/B 1525 ~/SCALA/runs/scala-gb10-v2/log.jsonl" 2>/dev/null | tail -1
