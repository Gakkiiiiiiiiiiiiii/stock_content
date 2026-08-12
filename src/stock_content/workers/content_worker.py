"""Worker entry point reserved for content-owned task execution."""


def main() -> None:
    raise SystemExit("No content task backend is configured; migrate repositories before starting the worker.")


if __name__ == "__main__":
    main()
