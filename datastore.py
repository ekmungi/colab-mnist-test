"""The data store: where a dataset comes from, in one of two modes chosen per dataset.

fetch: public data that is easy to download (torchvision, Hugging Face, an open link). The session downloads it onto the
       machine's own disk and nothing is kept in the artifacts folder, so it costs no Drive space.
store: data that needs a login, or that was kept on purpose. It sits once per project in <artifacts>/data/<name>/ (or as
       <name>.zip), synced with Drive, with a source.md note recording its origin and a checksum for every file. Each
       session copies it onto the machine's disk and unpacks a zip there, because the disk reads far faster than Drive.

A run never fills the data folder by itself, and credentials are never stored: a dataset behind a login is downloaded by
the person who holds the login, and keep_copy() is used only after that person says to keep a fetched dataset.
"""
import datetime
import hashlib
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")
MODES = ("fetch", "store")
MANIFEST_NAME = "source.md"
READY_MARKER = ".unpacked"
FETCHED_MARKER = ".fetched"
_MARKERS = (MANIFEST_NAME, READY_MARKER, FETCHED_MARKER)
_FILE_LINE = re.compile(r"^- (?P<name>.+?)  sha256=(?P<digest>[0-9a-f]{64})$")


class DatasetMissing(RuntimeError):
    """Raised when a dataset is not where its mode says; the message says how to add it."""


class ChecksumMismatch(RuntimeError):
    """Raised when an archive's checksum differs from the expected one."""


def sha256_of(path):
    """Returns the hex SHA-256 of a file, read in 1 MB pieces so large archives do not fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def find_archive(name, data_root):
    """Returns the archive file for dataset `name` in data_root, or None."""
    for suffix in ARCHIVE_SUFFIXES:
        candidate = Path(data_root) / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _inside(target, member_name):
    """Checks that an archive member lands inside `target`; raises ValueError when it would escape."""
    destination = (target / member_name).resolve()
    if target.resolve() not in destination.parents and destination != target.resolve():
        raise ValueError(f"archive member {member_name!r} would be written outside the target folder")


def _unpack(archive, target):
    """Unpacks a zip or tar archive into target, refusing any member that escapes it or is a link."""
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                _inside(target, member)
            bundle.extractall(target)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                _inside(target, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"archive member {member.name!r} is a link, which is not allowed")
            bundle.extractall(target)


def ensure_local_copy(name, data_root, work_dir, expected_sha256=None):
    """Returns a readable folder for a kept dataset, unpacking an archive into work_dir when there is one.

    Raises DatasetMissing when it is not there (it never downloads), ChecksumMismatch for a wrong archive, and
    ValueError for an archive that would write outside work_dir.
    """
    data_root, work_dir = Path(data_root), Path(work_dir)
    archive = find_archive(name, data_root)
    if archive is not None:
        if expected_sha256 is not None and sha256_of(archive) != expected_sha256:
            raise ChecksumMismatch(f"{archive.name} does not match the expected checksum")
        target = work_dir / name
        if not (target / READY_MARKER).exists():
            _unpack(archive, target)
            (target / READY_MARKER).write_text("unpacked\n")
        return target
    folder = data_root / name
    if folder.is_dir() and any(folder.iterdir()):
        return folder
    raise DatasetMissing(
        f"dataset {name!r} is set to store mode but is not in {data_root}. Put it there once (a person downloads it if it "
        f"needs a login, or keep_copy() saves a fetched one); it is never downloaded during a run in store mode.")


def resolve(name, mode, data_root, work_dir, fetch=None, expected_sha256=None):
    """Returns a readable folder for dataset `name` under its mode.

    store: the kept copy in data_root (an archive is unpacked into work_dir); never calls fetch.
    fetch: fetch(folder) downloads into work_dir/name once; data_root is never written.
    """
    if mode not in MODES:
        raise ValueError(f"data mode {mode!r} must be one of {MODES}")
    if mode == "store":
        return ensure_local_copy(name, data_root, work_dir, expected_sha256)
    if fetch is None:
        raise ValueError("fetch mode needs a fetch function")
    target = Path(work_dir) / name
    if not (target / FETCHED_MARKER).exists():
        fetch(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / FETCHED_MARKER).write_text("fetched\n")
    return target


def _files_under(folder):
    """Returns the dataset's files as sorted relative POSIX names, leaving out the manifest and the markers."""
    names = [p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file()]
    return sorted(n for n in names if n not in _MARKERS)


def _manifest_target(target):
    """Returns (manifest path, base folder, file names) for a dataset folder or a single archive file."""
    target = Path(target)
    if target.is_dir():
        return target / MANIFEST_NAME, target, _files_under(target)
    stem = target.name
    for suffix in ARCHIVE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return target.with_name(f"{stem}.{MANIFEST_NAME}"), target.parent, [target.name]


def write_manifest(target, source, licence, acquired_by, credentials_needed=False, today=None):
    """Writes the source note for a dataset folder (<folder>/source.md) or an archive (<name>.source.md).

    It records the origin, licence, date, who acquired it and a SHA-256 for every file. Never put a username, password
    or token in `source`; say only that credentials were needed.
    """
    manifest, base, names = _manifest_target(target)
    day = today or datetime.date.today().isoformat()
    login = "yes (held by the person who downloaded it, never stored)" if credentials_needed else "no"
    lines = [f"# {Path(target).name}", "", f"- source: {source}", f"- licence: {licence}", f"- acquired: {day} by {acquired_by}",
             f"- credentials needed: {login}", "", "## Files", ""]
    lines += [f"- {name}  sha256={sha256_of(base / name)}" for name in names]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_manifest(target):
    """Returns a list of problems (empty when every recorded file is present and unchanged)."""
    manifest, base, _ = _manifest_target(target)
    if not manifest.is_file():
        return [f"{manifest.name} is missing"]
    problems = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = _FILE_LINE.match(line)
        if not match:
            continue
        path = base / match["name"]
        if not path.is_file():
            problems.append(f"{match['name']} is missing")
        elif sha256_of(path) != match["digest"]:
            problems.append(f"{match['name']} has changed since it was recorded")
    return problems


def keep_copy(name, work_dir, data_root, source, licence, acquired_by, credentials_needed=False, archive=False, today=None):
    """Keeps a fetched dataset by copying it into the data folder, with its source note; use only when a person says to.

    With archive=True the copy is one zip file (smaller to store). Returns the new folder or zip path. Raises
    DatasetMissing when the dataset was never fetched and FileExistsError when a kept copy already exists.
    """
    source_dir = Path(work_dir) / name
    if not source_dir.is_dir():
        raise DatasetMissing(f"dataset {name!r} has not been fetched into {work_dir}, so there is nothing to keep")
    data_root = Path(data_root)
    destination = data_root / (f"{name}.zip" if archive else name)
    if destination.exists() or find_archive(name, data_root) or (data_root / name).exists():
        raise FileExistsError(f"a kept copy of {name!r} already exists in {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    if archive:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
            for relative in _files_under(source_dir):
                bundle.write(source_dir / relative, relative)
    else:
        shutil.copytree(source_dir, destination, ignore=shutil.ignore_patterns(*_MARKERS))
    write_manifest(destination, source, licence, acquired_by, credentials_needed, today)
    return destination
