# Thin wrapper around tasks.py so `make` and `python tasks.py` never diverge.
# tasks.py is the source of truth and is the supported entry point on Windows.

PY ?= python

.DEFAULT_GOAL := help
.PHONY: help setup lock fmt lint typecheck test test-unit test-integration \
        test-security security build docker-build run clean

help:
	@$(PY) tasks.py --list

setup lock fmt lint typecheck test test-unit test-integration test-security \
security build docker-build run clean:
	@$(PY) tasks.py $@
