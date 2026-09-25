.PHONY: all clean coverage coverage-open coverage-release coverage-release-open mutants quality install-deps setup-hooks

# Detect operating system to handle executable suffixes correctly (.exe on Windows)
ifeq ($(OS),Windows_NT)
    EXE_SUFFIX := .exe
    COPY := copy /Y
    FIX_PATH = $(subst /,\, $1)
else
    EXE_SUFFIX :=
    COPY := cp
    FIX_PATH = $1
endif

# OpenBench passes the output path via the EXE variable (e.g., EXE=whale-master)
# Default to "whale" if not specified
EXE ?= whale$(EXE_SUFFIX)

all:
	cargo build --release
	$(COPY) $(call FIX_PATH,target/release/whale$(EXE_SUFFIX)) $(call FIX_PATH,$(EXE))

clean:
	cargo clean

install-deps: setup-hooks
	rustup component add clippy rustfmt llvm-tools-preview
	cargo install cargo-llvm-cov
	cargo install cargo-mutants

setup-hooks:
	git config --local core.hooksPath .githooks
ifneq ($(OS),Windows_NT)
	chmod +x .githooks/pre-push
endif


# Non OpenBench
coverage:
	cargo llvm-cov --lib --html

coverage-open:
	cargo llvm-cov --lib --open

coverage-release:
	cargo llvm-cov --lib --release --html

coverage-release-open:
	cargo llvm-cov --lib --release --open

mutants:
	cargo mutants -- --lib

quality:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	RUST_MIN_STACK=16777216 cargo test --lib
	RUST_MIN_STACK=16777216 cargo test --tests --release
	RUST_MIN_STACK=16777216 cargo llvm-cov --lib --html --fail-under-lines 90
