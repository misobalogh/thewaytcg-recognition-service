"""Render numbered card PDFs from ZIP archives or data/pdf into reference PNGs."""

import argparse
from pathlib import Path
import zipfile

import pymupdf


ROOT = Path(__file__).resolve().parents[1]


def render_pdf(contents: bytes, target: Path, max_dim: int) -> None:
    with pymupdf.open(stream=contents, filetype="pdf") as document:
        if len(document) != 1:
            raise ValueError(f"Expected one card per PDF: {target.stem} has {len(document)} pages")
        page = document[0]
        scale = max_dim / max(page.rect.width, page.rect.height)
        page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "data/gt/png")
    parser.add_argument("--max-dim", type=int, default=1400)
    args = parser.parse_args()
    if args.max_dim < 64:
        parser.error("--max-dim must be at least 64")
    expected = {p.stem for p in (args.data_dir / "json").glob("*.json")}
    if not expected:
        raise ValueError("Missing data/json card metadata; run scripts.csv_to_json_schema first")
    sources = {}
    for path in sorted((args.data_dir / "pdf").glob("*.pdf")):
        if path.stem in expected:
            sources[path.stem] = (path, None)
    for archive in sorted(args.data_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.namelist():
                name = Path(entry)
                if name.suffix.lower() != ".pdf" or name.stem not in expected:
                    continue
                if name.stem in sources:
                    raise ValueError(f"Duplicate reference PDF ID: {name.stem}")
                sources[name.stem] = (archive, entry)
    missing = expected - set(sources)
    if missing:
        raise ValueError(f"Missing reference PDFs for IDs: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for card_id, (source, entry) in sorted(sources.items()):
        target = args.output / f"{card_id}.png"
        if entry is None:
            contents = source.read_bytes()
        else:
            with zipfile.ZipFile(source) as zipped:
                contents = zipped.read(entry)
        render_pdf(contents, target, args.max_dim)
        print(f"Rendered {target.name}", flush=True)
    print(f"Prepared {len(sources)} reference cards in {args.output}")


if __name__ == "__main__":
    main()
