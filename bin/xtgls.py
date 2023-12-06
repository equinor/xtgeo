"""xtgls: a small script for listing xtg formatted files."""
import argparse
import json
import pathlib

# import yaml
from struct import unpack

VALIDEXT = [".xtgregsurf", ".xtgregcube", ".xtgcpgeom"]


def extract_meta_regsurf(fil):
    """Extract metadata for RegularSurface."""
    #
    offset = 28
    with open(fil, "rb") as stream:
        buf = stream.read(offset)
    # unpack header
    _, _, nfloat, ncol, nrow = unpack("= i i i q q", buf)

    # now jump to metadata:
    pos = offset + nfloat * ncol * nrow + 13

    with open(fil, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()
    return json.loads(jmeta, object_pairs_hook=dict)


def extract_meta_regcube(fil):
    """Extract metadata for RegularCube."""
    #
    offset = 36
    with open(fil, "rb") as stream:
        buf = stream.read(offset)
    # unpack header
    _, _, nfloat, ncol, nrow, nlay = unpack("= i i i q q q", buf)

    # now jump to metadata:
    pos = offset + nfloat * ncol * nrow * nlay + 13

    with open(fil, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()
    return json.loads(jmeta, object_pairs_hook=dict)


def extract_meta_cpgeom(fil):
    """Extract metadata for Corner Point Geometry."""
    #
    offset = 36
    with open(fil, "rb") as stream:
        buf = stream.read(offset)
    # unpack header
    _, _, nfloat, ncol, nrow, nlay = unpack("= i i i q q q", buf)
    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    nc, nz, na = [int(val) for val in str(nfloat)]

    pos = offset + nc * nncol * nnrow * 6 + 13
    pos += nz * nncol * nnrow * nnlay * 4
    pos += na * ncol * nrow * nlay

    with open(fil, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()
    return json.loads(jmeta, object_pairs_hook=dict)


def extract_meta(fil):
    """Read first bytes to identify file format."""
    with open(fil, "rb") as stream:
        buf = stream.read(8)
    # unpack header
    swap, magic = unpack("= i i", buf)

    res = None
    typ = "Undefined"
    if swap == 1 and magic == 1101:
        res = extract_meta_regsurf(fil)
        typ = "Regular Surface"

    if swap == 1 and magic == 1201:
        res = extract_meta_regcube(fil)
        typ = "Regular Cube"

    if swap == 1 and magic == 1301:
        res = extract_meta_cpgeom(fil)
        typ = "Corner Point Geometry"

    return res, typ


def collect_files(infiles):
    """List all files."""
    files = []
    for fil in infiles:
        fil = pathlib.Path(fil)
        if fil.suffix in VALIDEXT:
            files.append(fil)
    return files


def print_meta(fil, res, args, typ):
    """Different ways of showing result."""
    fil = str(fil)
    if args.prettify:
        print(f"\n{80 * '='}\n{fil}: {typ}\n")
        print(json.dumps(res, indent=4))
    else:
        print(f"{fil} {typ}: {res}")


def show_meta(fil, res, args, typ):
    """Show meta."""
    if args.required:
        res = {"_required_": res["_required_"]}
    elif args.optional:
        res = {"_optional_": res["_optional_"]}

    print_meta(fil, res, args, typ)


def main(args):
    """Main function for entry point."""
    usefiles = collect_files(args.files)
    for fil in usefiles:
        res, typ = extract_meta(fil)

        show_meta(fil, res, args, typ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs="+", help="Files e.g. *")
    parser.add_argument(
        "--required", action="store_true", help="Only show _required_ metadata"
    )
    parser.add_argument(
        "--optional", action="store_true", help="Only show _optional_ metadata"
    )
    parser.add_argument("--prettify", action="store_true", help="Show as pretty")

    args = parser.parse_args()
    main(args)
