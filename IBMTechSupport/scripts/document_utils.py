from pathlib import Path


def chunk_filename(chunk):
    metadata = chunk.get("metadata")

    targets = (
        [chunk, metadata]
        if isinstance(metadata, dict)
        else [chunk]
    )

    for data in targets:
        for key in ("filename", "source_file", "source", "document_id", "doc_id"):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value

    raise ValueError(f"Missing chunk filename: {list(chunk)}")


def filename_key(filename):
    return (str(filename).strip().replace("\\", "/").split("/")[-1].casefold())


def normalize_filename(filename):
    return Path(filename.replace("\\", "/")).name.casefold()